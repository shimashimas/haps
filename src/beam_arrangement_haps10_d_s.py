#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
beam_arrangement_single_disaster.py
(災害エリア中心へのシングルビーム配置)

変更点:
- 格子配置を廃止し、DISASTER_CENTER_X/Y_KM の地点にビームを1つだけ配置
- 彩色は1色固定(cl1)
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

# ==================== 出力先 ====================
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "database" / "beamlist_single"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 基本設定 ====================
HAPS_LOCATION = {'lat': 26.4816, 'lon': 127.9755}   # 宜野座
SERVICE_RADIUS_KM = 50.0

# ---- 指定：ビーム半径18km ----
BEAM_RADIUS_KM = 18.0

# ---- 災害エリア中心設定 ----
DISASTER_ENABLE = True
# HAPS中心を(0,0)とした相対座標(km) x=東, y=北
DISASTER_CENTER_X_KM = 21.0
DISASTER_CENTER_Y_KM = 25.0
DISASTER_RADIUS_KM   = 18.0 # ここでは参考値（ビーム半径と同じに設定）

COLOR_LIST = {"cl1": "#4FAAD1", "cl2": "#EBBF00", "cl3": "#B66427"}

# ---- sat_radius算出用 ----
HAPS_ALT_KM = 20.0
CARRIER_FREQ_HZ = 2.0e9
APERTURE_K_DEG = 70.0
USE_APERTURE_CAP = True
APERTURE_MAX_M = 1.0

LAND_GEOJSON_PATH = "japan.geojson"
PROJ_CRS = "EPSG:32652" # UTM Zone 52N

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
    c = 299792458.0
    lam = c / f_hz
    hpbw_deg = 2.0 * np.degrees(np.arctan2(r_km, h_km))
    if hpbw_deg <= 0:
        return 0.05
    D = (k_deg * lam) / hpbw_deg
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

def latlon_to_xy_km(lat, lon):
    dx = (lon - HAPS_LOCATION["lon"]) * km_per_deg_lon(HAPS_LOCATION["lat"])
    dy = (lat - HAPS_LOCATION["lat"]) * km_per_deg_lat()
    return float(dx), float(dy)

# ★追加: 相対座標(km)から緯度経度へ変換
def xy_km_to_latlon(x_km, y_km):
    lat_c = HAPS_LOCATION["lat"]
    lon_c = HAPS_LOCATION["lon"]
    
    # latlon_to_xy_km の逆算
    # dy = (lat - lat_c) * 111.0  => lat = lat_c + dy / 111.0
    lat = lat_c + y_km / km_per_deg_lat()
    
    # dx = (lon - lon_c) * 111.0 * cos(lat_c) => lon = lon_c + dx / (...)
    lon = lon_c + x_km / km_per_deg_lon(lat_c)
    
    return float(lat), float(lon)

# ==================== シングルビーム配置 ====================
beam_center = []   # list of dict {lat, lon}
beam_radius = []   # list of float[km]
sat_radius = []    # list of float[m]

def setup_single_disaster_beam():
    print("災害エリア中心へのシングルビーム配置...")
    
    # 中心座標(km)を緯度経度に変換
    lat, lon = xy_km_to_latlon(DISASTER_CENTER_X_KM, DISASTER_CENTER_Y_KM)
    
    R = float(BEAM_RADIUS_KM)
    
    # 1つだけ追加
    beam_center.append({"lat": lat, "lon": lon})
    beam_radius.append(R)
    sat_radius.append(calc_sat_radius_m(R))
    
    print(f"配置完了: Center(lat={lat:.4f}, lon={lon:.4f}), R={R}km")

# ==================== 人口割り当て（1ビーム用） ====================
def calculate_users_fast():
    print("人口カバー計算...")
    if len(beam_center) == 0:
        return []

    df = pref_list.copy()
    # 必要な列チェック
    for col in ["緯度", "経度", "人口"]:
        if col not in df.columns:
            return [0] * len(beam_center)

    df["人口"] = pd.to_numeric(df["人口"], errors="coerce").fillna(0).astype(int)
    df = df[df["人口"] > 0].copy()
    if df.empty:
        return [0] * len(beam_center)

    pts = gpd.GeoDataFrame(
        df,
        geometry=[Point(lon, lat) for lat, lon in zip(df["緯度"], df["経度"])],
        crs="EPSG:4326"
    ).to_crs(PROJ_CRS)

    beams = gpd.GeoDataFrame(
        {"bid": np.arange(len(beam_center), dtype=int)},
        geometry=[Point(b["lon"], b["lat"]) for b in beam_center],
        crs="EPSG:4326"
    ).to_crs(PROJ_CRS)

    r_m = float(BEAM_RADIUS_KM) * 1000.0
    beams["geometry"] = beams.geometry.buffer(r_m)

    # 結合 (1ビームなので処理は軽い)
    joined = gpd.sjoin(pts, beams, how="inner", predicate="within")

    if joined.empty:
        print("人口エリア外です。User=0")
        return [0]

    # 集計
    user_sum = joined["人口"].sum()
    print(f"利用者数計算完了: {user_sum} 人")
    return [int(user_sum)]

# ==================== 出力 ====================
beam_param = []

def create_beam_param_list(user_counts):
    beam_param.clear()
    # シングルビームなのでループは1回、色は固定(1)
    for i in range(len(beam_center)):
        pos = beam_center[i]
        beam_param.append([
            pos["lat"],
            pos["lon"],
            float(BEAM_RADIUS_KM),
            1,                  # color 1固定
            int(user_counts[i]),
            float(sat_radius[i]),
        ])

def output_pic(tag):
    print("地図画像を出力中...")
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_map.plot(ax=ax, edgecolor="#444", facecolor="white", linewidth=0.5)

    lat_c, lon_c = HAPS_LOCATION["lat"], HAPS_LOCATION["lon"]
    # 表示範囲調整
    ##view_km = max(SERVICE_RADIUS_KM, abs(DISASTER_CENTER_X_KM), abs(DISASTER_CENTER_Y_KM)) + BEAM_RADIUS_KM + 10
    view_km = SERVICE_RADIUS_KM * 1.1
    lat_delta = view_km / km_per_deg_lat()
    lon_delta = view_km / km_per_deg_lon(lat_c)
    ax.set_xlim([lon_c - lon_delta, lon_c + lon_delta])
    ax.set_ylim([lat_c - lat_delta, lat_c + lat_delta])

    # HAPSサービス円(参考)
    service_r_deg = SERVICE_RADIUS_KM / km_per_deg_lon(lat_c)
    ax.add_patch(patches.Circle((lon_c, lat_c), service_r_deg, fill=False, ls="--", ec="red", lw=1.2, label="HAPS Service Area"))

    # ターゲット座標(災害中心)の十字マーク
    cx_lat, cx_lon = xy_km_to_latlon(DISASTER_CENTER_X_KM, DISASTER_CENTER_Y_KM)
    ##ax.plot(cx_lon, cx_lat, 'x', color='magenta', markersize=10, markeredgewidth=2, label="Target Center")

    # ビーム描画
    for lat, lon, rad_km, color_idx, _user, _sat in beam_param:
        r_deg = rad_km / km_per_deg_lon(lat)
        ax.add_patch(
            patches.Circle(
                (lon, lat),
                r_deg,
                color=COLOR_LIST.get(f"cl{int(color_idx)}", "#888888"),
                alpha=0.45,
                label=f"Single Beam (User={_user})"
            )
        )

    # 凡例整理（重複排除）
    ##handles, labels = ax.get_legend_handles_labels()
    ##by_label = dict(zip(labels, handles))
    ##ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    ax.set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3)
    plt.title(f"Single Beam Deployment (Target: {DISASTER_CENTER_X_KM},{DISASTER_CENTER_Y_KM} km)", fontsize=12)
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
    tag = f"single_r{BEAM_RADIUS_KM}km_at_{DISASTER_CENTER_X_KM}_{DISASTER_CENTER_Y_KM}"

    print(f"--- HAPSシングルビーム配置 ---")
    print(f"Target (km): x={DISASTER_CENTER_X_KM}, y={DISASTER_CENTER_Y_KM}")
    print(f"Beam Radius: {BEAM_RADIUS_KM} km")
    
    # 1. ビーム配置（1つだけ生成）
    setup_single_disaster_beam()
    
    # 2. ユーザ数計算
    users = calculate_users_fast()
    
    # 3. リスト作成 (彩色は固定なので計算不要)
    create_beam_param_list(users)
    
    # 4. 出力
    output_pic(tag)
    output_csv(tag)

    print(f"処理時間 = {time.time() - t0:.2f} 秒")

if __name__ == "__main__":
    main()