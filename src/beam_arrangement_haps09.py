#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
beam_arrangement_haps09.py
(3色 / 六角格子(三角格子) / 人口+有人島優先 / 円交差で隣接 / sat_radius(開口半径)を配置側で算出)

主な追加・変更
- ビーム間隔を PITCH_RATIO で調整（中心間距離 = 2R * PITCH_RATIO）
- 人口割当を GeoPandas sjoin で高速化（点→円の一括判定）
- sat_radius[m] を beam_radius[km] から算出してCSVに追加出力
  （シミュレーション側はCSVの sat_radius を優先利用する想定）

出力
- HAPS_beam_list_Okinawa_r{R}km_p{PITCH_RATIO}.csv
  columns: latitude, longitude, beam_radius, color, user, sat_radius
- HAPS_beams_Okinawa_r{R}km_p{PITCH_RATIO}.png
"""
from pathlib import Path

import os, sys, time, csv
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Point

# ==================== 出力先（この .py からの相対パス） ====================
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "database" / "beamlist_p150"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 基本設定 ====================
HAPS_LOCATION = {'lat': 26.4816, 'lon': 127.9755}   # 宜野座
SERVICE_RADIUS_KM = 50.0

# ---- ここを振る ----
BEAM_RADIUS_KM = 5.0

# ---- 追加：ビーム間隔の比率（重ね具合）----
# 中心間距離 = 2 * R * PITCH_RATIO
# 1.0: ほぼ接する, 0.9: 重ねる, 1.1: 離す
PITCH_RATIO = 1.0

COLOR_LIST = {"cl1": "#4FAAD1", "cl2": "#EBBF00", "cl3": "#B66427"}

# ---- sat_radius算出用（方針：配置側で決める）----
HAPS_ALT_KM = 20.0
CARRIER_FREQ_HZ = 2.0e9
APERTURE_K_DEG = 70.0
USE_APERTURE_CAP = True
APERTURE_MAX_M = 1.0  # capするなら上限

LAND_GEOJSON_PATH = "japan.geojson"                 # 陸地ポリゴン（無くてもOK）
PROJ_CRS = "EPSG:32652"                             # UTM Zone 52N（沖縄向け）

POP_CSV = "database\\jinko_list_sityoson.csv"
MAP_GEOJSON = "japan_eez.json"


# ==================== 入力チェック ====================
t0 = time.time()
if not os.path.exists(POP_CSV):
    print(f"エラー: 人口データ '{POP_CSV}' が見つかりません。")
    sys.exit(1)
if not os.path.exists(MAP_GEOJSON):
    print(f"エラー: 地図データ '{MAP_GEOJSON}' が見つかりません。")
    sys.exit(1)

try:
    pref_list = pd.read_csv(POP_CSV, encoding="utf-8")
except UnicodeDecodeError:
    pref_list = pd.read_csv(POP_CSV, encoding="shift-jis")

gdf_map = gpd.read_file(MAP_GEOJSON)


# ==================== ユーティリティ ====================
def aperture_radius_from_ground_radius(r_km, h_km, f_hz, k_deg=70.0):
    """
    地表ビーム半径 r_km から -3dB開口半径 a[m] を求める（経験式）
    """
    c = 299792458.0
    lam = c / f_hz
    hpbw_deg = 2.0 * np.degrees(np.arctan2(r_km, h_km))  # -3dBビーム幅（概算）
    if hpbw_deg <= 0:
        return 0.05
    D = (k_deg * lam) / hpbw_deg  # 開口径[m]
    a = max(D / 2.0, 0.01)
    return a

def calc_sat_radius_m(r_km):
    a = aperture_radius_from_ground_radius(r_km, HAPS_ALT_KM, CARRIER_FREQ_HZ, APERTURE_K_DEG)
    if USE_APERTURE_CAP:
        a = min(a, APERTURE_MAX_M)
    return float(a)

def km_per_deg_lat():
    return 111.0

def km_per_deg_lon(lat_deg):
    return 111.0 * np.cos(np.radians(lat_deg))


# ==================== 配置（PITCH_RATIOつき三角格子） ====================
beam_center = []   # list of dict {lat, lon}
beam_radius = []   # list of float[km]
sat_radius = []    # list of float[m]

def within_service(lat, lon):
    # 速度より確実性優先：大雑把に緯度経度→km換算で円判定（中心近傍なら十分）
    dx = (lon - HAPS_LOCATION["lon"]) * km_per_deg_lon(HAPS_LOCATION["lat"])
    dy = (lat - HAPS_LOCATION["lat"]) * km_per_deg_lat()
    return (dx*dx + dy*dy) <= (SERVICE_RADIUS_KM*SERVICE_RADIUS_KM)

def setup_triangular_lattice():
    """
    三角格子（=六角最密）で中心配置。
    - 隣接中心距離: pitch_km = 2R * PITCH_RATIO
    - 行間隔: row_step = sqrt(3)/2 * pitch
    - 偶奇行で半ピッチずらし
    """
    print("三角格子でビーム初期配置...")
    lat_c, lon_c = HAPS_LOCATION["lat"], HAPS_LOCATION["lon"]
    R = float(BEAM_RADIUS_KM)

    pitch_km = 2.0 * R * float(PITCH_RATIO)
    row_step_km = (np.sqrt(3.0) / 2.0) * pitch_km

    search_km = SERVICE_RADIUS_KM + R
    lat_span = search_km / km_per_deg_lat()
    lon_span = search_km / km_per_deg_lon(lat_c)

    lat_step = row_step_km / km_per_deg_lat()
    lon_step = pitch_km / km_per_deg_lon(lat_c)

    lat_range = np.arange(lat_c - lat_span, lat_c + lat_span + 1e-12, lat_step)
    lon_range = np.arange(lon_c - lon_span, lon_c + lon_span + 1e-12, lon_step)

    for i, lat in enumerate(lat_range):
        for lon in lon_range:
            offset = (0.5 * lon_step) if (i % 2 == 1) else 0.0
            lon2 = lon + offset
            if within_service(float(lat), float(lon2)):
                beam_center.append({"lat": float(lat), "lon": float(lon2)})
                beam_radius.append(R)
                sat_radius.append(calc_sat_radius_m(R))

    print(f"HAPSエリア内の初期ビーム数: {len(beam_center)}")


# ==================== 人口割り当て（高速：sjoinで一括） ====================
def calculate_users_fast():
    """
    人口点（自治体代表点）を beams の円に sjoin して
    - 1点が複数ビームに入る場合、人口を等分配
    を一括で処理する。
    """
    print("人口カバー計算（高速）...")

    if len(beam_center) == 0:
        return []

    df = pref_list.copy()
    # 必須列が想定通りでない場合に備えて最低限チェック
    for col in ["緯度", "経度", "人口"]:
        if col not in df.columns:
            raise RuntimeError(f"人口CSVに列 '{col}' がありません: columns={list(df.columns)}")

    df["人口"] = pd.to_numeric(df["人口"], errors="coerce").fillna(0).astype(int)
    df = df[df["人口"] > 0].copy()
    if df.empty:
        print("人口>0のデータがありません。")
        return [0] * len(beam_center)

    # ポイント作成（WGS84→UTM）
    pts = gpd.GeoDataFrame(
        df.reset_index(drop=True),
        geometry=[Point(lon, lat) for lat, lon in zip(df["緯度"], df["経度"])],
        crs="EPSG:4326"
    ).to_crs(PROJ_CRS)
    pts["pid"] = np.arange(len(pts), dtype=int)

    # ビーム円作成（WGS84→UTM）
    beams = gpd.GeoDataFrame(
        {"bid": np.arange(len(beam_center), dtype=int),
         "lat": [b["lat"] for b in beam_center],
         "lon": [b["lon"] for b in beam_center]},
        geometry=[Point(b["lon"], b["lat"]) for b in beam_center],
        crs="EPSG:4326"
    ).to_crs(PROJ_CRS)

    r_m = float(BEAM_RADIUS_KM) * 1000.0
    beams["geometry"] = beams.geometry.buffer(r_m)

    # 空間結合（点が円の中に入る）
    joined = gpd.sjoin(
        pts[["pid", "人口", "geometry"]],
        beams[["bid", "geometry"]],
        how="inner",
        predicate="within"
    )

    if joined.empty:
        print("人口点がどのビームにも入っていません。")
        return [0] * len(beam_center)

    # 1点あたり何ビームにカバーされたか
    cover_cnt = joined.groupby("pid")["bid"].count().rename("ncover")
    joined = joined.join(cover_cnt, on="pid")

    # 人口を等分配して各ビームに加算
    joined["share"] = joined["人口"] / joined["ncover"]
    beam_user = joined.groupby("bid")["share"].sum()

    out = np.zeros(len(beam_center), dtype=float)
    out[beam_user.index.values] = beam_user.values
    out = np.rint(out).astype(int).tolist()

    print("利用者数計算完了")
    return out


# ==================== 有人島抽出（従来踏襲） ====================
def load_inhabited_land_polygons():
    if not os.path.exists(LAND_GEOJSON_PATH):
        print(f"注意: 陸地 '{LAND_GEOJSON_PATH}' が見つからないため、ユーザ数>0のみで継続。")
        return None
    try:
        land_gdf = gpd.read_file(LAND_GEOJSON_PATH)
    except Exception as e:
        print(f"注意: 陸地ファイル読込失敗: {e}（スキップ）。")
        return None

    df = pref_list.copy()
    df["人口"] = pd.to_numeric(df["人口"], errors="coerce").fillna(0)
    df = df[df["人口"] > 0].copy()
    if df.empty:
        print("注意: 人口>0の自治体が見つからず、有人島抽出をスキップ。")
        return None

    gpts = gpd.GeoDataFrame(
        df, geometry=[Point(lon, lat) for lat, lon in zip(df["緯度"], df["経度"])], crs="EPSG:4326"
    )
    try:
        joined = gpd.sjoin(land_gdf.to_crs("EPSG:4326"), gpts, how="inner", predicate="intersects")
    except TypeError:
        joined = gpd.sjoin(land_gdf.to_crs("EPSG:4326"), gpts, how="inner", op="intersects")

    if joined.empty:
        print("注意: 有人島ポリゴンが検出できず、スキップ。")
        return None

    inhabited_ids = joined.index.unique()
    inhabited = land_gdf.loc[inhabited_ids].copy().to_crs("EPSG:4326")

    # サービス範囲付近に絞る
    lat_c, lon_c = HAPS_LOCATION["lat"], HAPS_LOCATION["lon"]
    span_km = SERVICE_RADIUS_KM + BEAM_RADIUS_KM
    lat_span = span_km / km_per_deg_lat()
    lon_span = span_km / km_per_deg_lon(lat_c)
    bbox = (lon_c - lon_span, lat_c - lat_span, lon_c + lon_span, lat_c + lat_span)
    inhabited = inhabited.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    if inhabited.empty:
        print("注意: サービス範囲内の有人島が無く、スキップ。")
        return None
    return inhabited

def beams_cover_inhabited_land(inhabited_gdf):
    keep = set()
    if inhabited_gdf is None or not beam_center:
        return keep

    bc = gpd.GeoDataFrame(
        {"bid": np.arange(len(beam_center), dtype=int)},
        geometry=[Point(b["lon"], b["lat"]) for b in beam_center],
        crs="EPSG:4326"
    ).to_crs(PROJ_CRS)

    land_m = inhabited_gdf.to_crs(PROJ_CRS)
    r_m = float(BEAM_RADIUS_KM) * 1000.0
    bc["circle"] = bc.geometry.buffer(r_m)

    # 円と有人島の交差
    for i, row in bc.iterrows():
        if land_m.intersects(row["circle"]).any():
            keep.add(int(row["bid"]))
    return keep


# ==================== 隣接（円どうしの intersects） ====================
def build_adjacency_geometric(active_indices):
    if not active_indices:
        return {}

    bc = gpd.GeoDataFrame(
        {"bid": active_indices,
         "lat": [beam_center[i]["lat"] for i in active_indices],
         "lon": [beam_center[i]["lon"] for i in active_indices]},
        geometry=[Point(beam_center[i]["lon"], beam_center[i]["lat"]) for i in active_indices],
        crs="EPSG:4326"
    ).to_crs(PROJ_CRS)

    r_m = float(BEAM_RADIUS_KM) * 1000.0 * 1.21
    circles = bc.copy()
    circles["geometry"] = bc.geometry.buffer(r_m)
    sindex = circles.sindex

    adj = {i: [] for i in active_indices}
    # circles の行番号は 0..len-1 なので、bidへの参照は circles.iloc[pos]["bid"] を使う
    for a_pos, a_row in circles.reset_index(drop=True).iterrows():
        Ai = int(a_row["bid"])
        cand = list(sindex.intersection(a_row.geometry.bounds))
        for b_pos in cand:
            if b_pos <= a_pos:
                continue
            Bj = int(circles.reset_index(drop=True).iloc[b_pos]["bid"])
            if a_row.geometry.intersects(circles.reset_index(drop=True).iloc[b_pos].geometry):
                adj[Ai].append(Bj)
                adj[Bj].append(Ai)
    return adj


# ==================== 彩色（DSATUR + 検査 + フォールバック） ====================
def color_dsat(adj, palette=(1,2,3)):
    color = {}
    nodes = sorted(adj.keys(), key=lambda n: len(adj[n]), reverse=True)
    sat = defaultdict(set)

    for _ in range(len(nodes)):
        uncolored = [n for n in nodes if n not in color]
        if not uncolored:
            break
        pick = max(uncolored, key=lambda n: (len(sat[n]), len(adj[n])))
        used = {color.get(nb) for nb in adj[pick] if nb in color}
        for c in palette:
            if c not in used:
                color[pick] = c
                break
        for nb in adj[pick]:
            sat[nb].add(color[pick])
    return color

def validate_coloring(adj, color):
    bad = []
    for u, nbrs in adj.items():
        for v in nbrs:
            if v <= u:
                continue
            if color.get(u) == color.get(v):
                bad.append((u, v))
    return (len(bad) == 0, bad)

def round_robin_coloring(nodes, palette=(1,2,3)):
    color = {}
    p = list(palette)
    k = 0
    for n in sorted(nodes):
        color[n] = p[k % len(p)]
        k += 1
    return color


# ==================== 出力 ====================
beam_param = []

def create_beam_param_list(user_counts, color_of, keep_indices):
    beam_param.clear()
    for i in sorted(keep_indices):
        pos = beam_center[i]
        beam_param.append([
            pos["lat"],
            pos["lon"],
            float(BEAM_RADIUS_KM),
            int(color_of.get(i, 1)),
            int(user_counts[i]),
            float(sat_radius[i]),   # <- 追加
        ])
    print(f"最終的なビーム数: {len(beam_param)}")

def output_pic(tag):
    print("地図画像を出力中...")
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_map.plot(ax=ax, edgecolor="#444", facecolor="white", linewidth=0.5)

    lat_c, lon_c = HAPS_LOCATION["lat"], HAPS_LOCATION["lon"]
    view_km = SERVICE_RADIUS_KM * 1.1
    lat_delta = view_km / km_per_deg_lat()
    lon_delta = view_km / km_per_deg_lon(lat_c)
    ax.set_xlim([lon_c - lon_delta, lon_c + lon_delta])
    ax.set_ylim([lat_c - lat_delta, lat_c + lat_delta])

    service_r_deg = SERVICE_RADIUS_KM / km_per_deg_lon(lat_c)
    ax.add_patch(patches.Circle((lon_c, lat_c), service_r_deg, fill=False, ls="--", ec="red", lw=1.2))

    for lat, lon, rad_km, color_idx, _user, _sat in beam_param:
        r_deg = rad_km / km_per_deg_lon(lat)
        ax.add_patch(
            patches.Circle(
                (lon, lat),
                r_deg,
                color=COLOR_LIST.get(f"cl{int(color_idx)}", "#888888"),
                alpha=0.45,
            )
        )

    ax.set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3)
    plt.title(f"HAPS Beam Arrangement (R={BEAM_RADIUS_KM} km, pitch=2R*{PITCH_RATIO})", fontsize=12)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    fn = OUT_DIR / f"HAPS_beamlist_{tag}.png"
    plt.savefig(fn, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"'{fn}' に出力しました。")


def output_csv(tag):
    fn = OUT_DIR / f"HAPS_beamlist_{tag}.csv"
    with open(fn, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["latitude", "longitude", "beam_radius", "color", "user", "sat_radius"])
        w.writerows(beam_param)
    print(f"'{fn}' に出力しました。")

# ==================== メイン ====================
def main():
    tag = f"r{BEAM_RADIUS_KM:g}km_p{PITCH_RATIO:g}"
    print(f"--- HAPSビーム配置（沖縄）: R={BEAM_RADIUS_KM} km, pitch=2R*{PITCH_RATIO}, Service={SERVICE_RADIUS_KM} km ---")
    print(f"sat_radius[m] (all beams) = {calc_sat_radius_m(BEAM_RADIUS_KM):.3f}  (cap={USE_APERTURE_CAP}, max={APERTURE_MAX_M})")

    setup_triangular_lattice()
    if not beam_center:
        print("HAPSエリア内に配置されるビームがありません。")
        return

    users = calculate_users_fast()

    inhabited = load_inhabited_land_polygons()
    land_keep = beams_cover_inhabited_land(inhabited) if inhabited is not None else set()

    candidate = sorted(set([i for i, u in enumerate(users) if u > 0]) | land_keep)
    if not candidate:
        print("人口・陸地のいずれでも採用条件を満たすビームがありませんでした。")
        return

    adj = build_adjacency_geometric(candidate)

    # 彩色
    if all(len(adj.get(n, [])) == 0 for n in candidate):
        color_of = round_robin_coloring(candidate, palette=(1, 2, 3))
    else:
        color_of = color_dsat(adj, palette=(1, 2, 3))
        ok, bad = validate_coloring(adj, color_of)
        if not ok:
            # 簡易グリーディ再彩色
            color_of = {}
            for n in sorted(candidate, key=lambda n: len(adj[n]), reverse=True):
                used = {color_of.get(nb) for nb in adj[n] if nb in color_of}
                for c in (1, 2, 3):
                    if c not in used:
                        color_of[n] = c
                        break
            ok2, bad2 = validate_coloring(adj, color_of)
            if not ok2:
                print("警告: 3色で完全分割に失敗。問題ペア:", bad2)

    create_beam_param_list(users, color_of, set(candidate))
    output_pic(tag)
    output_csv(tag)

    print(f"処理時間 = {time.time() - t0:.2f} 秒")


if __name__ == "__main__":
    main()
