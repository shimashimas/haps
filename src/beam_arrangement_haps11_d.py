# -*- coding: utf-8 -*-
"""
HAPS downlink simulation (best-server on ground grid) + coverage metrics

- Main: (A) constant FSPL using center slant range (fast)
- Option: (B) per-grid FSPL using slant range d(x,y) (slower but more realistic)

- Antenna aperture radius a[m] is provided by beam arrangement CSV as "sat_radius" (RECOMMENDED).
- Antenna pattern switchable: "BESSEL", "M2101", "TR38811"

- Throughput mapping switchable:
    1) CQI table method (LTE-like)
    2) Shannon + clipping method

Added metrics (inside evaluation mask):
  - coverage_rate
  - low_quality_rate_all
  - low_quality_rate_connected
  - unconnected_rate

Outputs (switchable):
  - result_*.csv
  - best_server_summary_*.csv
  - grid_quality_metrics_*.csv
  - run_log.csv (append per run, recommended)
  - (optional) throughput_map_*.png, sinr_map_*.png
  - (optional) maps_*.npz
"""

# ===================== imports =====================
import os
import csv
import time
import datetime

import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
from geopy.distance import geodesic


# ===================== 1) Config =====================
# ---- HAPS parameters ----
HAPS_LOCATION = {'lat': 26.4816, 'lon': 127.9755}  # example
HAPS_ALT_KM   = 20.0                               # [km]
CARRIER_FREQ  = 2.0e9                              # [Hz]

REPEATED_BEAM = 3  # 3-color reuseなら3
SERVICE_RADIUS_KM = 50.0                           # all評価用の円（CENTER基準）

TOTAL_BANDWIDTH = 20e6                             # [Hz] total (split by REPEATED_BEAM)
TOTAL_POWER     = 20.0                             # [W] total TX power (all beams)

# TX_GAIN_DB is treated as "peak gain excluding the normalized pattern gb(θ)" (gb peak = 1).
TX_GAIN_DB = 24.0                                  # [dBi]
RX_GAIN_DB = 0.0                                   # [dBi]
MISC_LOSS_DB = 0.0                                 # [dB]

# ---- Antenna pattern ----
ANT_PATTERN = "TR38811"        # "BESSEL", "M2101", "TR38811"
M2101_SLL_DB = 20              # 20 or 30
TR38811_MAX_ATTEN_DB = 30.0    # [dB] max attenuation for TR38811-like pattern

# ---- Aperture source policy ----
REQUIRE_SAT_RADIUS = True
SAT_RADIUS_FALLBACK_M = 0.35

# ---- Link budget option ----
USE_DISTANCE_FSPL = False

# ---- Throughput mapping ----
THROUGHPUT_MODE = "SHANNON"        # "CQI" or "SHANNON"
SHANNON_GAP_DB = 0.0
ETA_MAX = 6.0
ETA_MIN = 0.0

# CQI table (LTE-like). Replace thresholds with your adopted table when finalizing.
CQI_TABLE = [
    (1,  0.1523, -6.7),
    (2,  0.2344, -4.7),
    (3,  0.3770, -2.3),
    (4,  0.6016,  0.2),
    (5,  0.8770,  2.4),
    (6,  1.1758,  4.3),
    (7,  1.4766,  5.9),
    (8,  1.9141,  8.1),
    (9,  2.4063, 10.3),
    (10, 2.7305, 11.7),
    (11, 3.3223, 14.1),
    (12, 3.9023, 16.3),
    (13, 4.5234, 18.7),
    (14, 5.1152, 21.0),
    (15, 5.5547, 22.7),
]

# ---- Quality / Coverage metrics thresholds ----
CONNECT_SINR_MIN_DB = -6.7
LOW_QUALITY_SINR_DB = 0.0

# ---- Disaster evaluation (mask center offset) ----
DISASTER_ENABLE = True
DISASTER_EXPORT_ALL_TOO = True   # True: all + disaster を両方計算してrun_logへ
DISASTER_CENTER_X_KM = 21.0
DISASTER_CENTER_Y_KM = 25.0
DISASTER_RADIUS_KM   = 18.0

# ---- IO ----
INPUT_BEAM_CSV = r"database\beamlist_disaster\HAPS_beamlist_single_r50km.csv"
OUTPUT_DIR_BASE = r"Result\Shannon_TR38811\disaster"

# ---- Mesh ----
MESH_MAX_KM = 100.0
NUM_SAMPLES = 401

# ---- Outputs (master switches) ----
OUT_ENABLE = True
OUT_MAKE_DIR = True
OUT_PRINT = True

# Per-output switches
OUT_CSV_RESULT_SUMMARY = True
OUT_CSV_BEST_SERVER_SUM = True
OUT_CSV_GRID_METRICS = True

# Aggregate run log (append)
OUT_CSV_RUNLOG = True
RUNLOG_FILENAME = "run_log.csv"

# Plot control
OUT_PLOT_MAP = False
OUT_SAVE_MAP = False
OUT_SHOW_PLOT = False

# Optional large arrays save
OUT_SAVE_NUMPY = False


# ===================== 2) Utility =====================
def _p(msg: str):
    if OUT_PRINT:
        print(msg)

def to_dB(x):
    return 10.0 * np.log10(x + 1e-30)

def from_dB(dB_val):
    return 10.0 ** (dB_val / 10.0)

def fspl_db(d_m, f_hz):
    c = 299792458.0
    lam = c / f_hz
    return 20.0 * np.log10(4.0 * np.pi * d_m / lam + 1e-30)

# Noise
K_BOLTZ = 1.38064852e-23
UE_NF_DB = 3.0
T_SYS = 290.0 * (10 ** (UE_NF_DB / 10.0))

def calc_noise_mw(bandwidth_Hz):
    return K_BOLTZ * T_SYS * bandwidth_Hz * 1000.0

def make_circle_mask(mesh_x, mesh_y, center_x_km, center_y_km, radius_km):
    dx = mesh_x - float(center_x_km)
    dy = mesh_y - float(center_y_km)
    return (dx * dx + dy * dy) <= (float(radius_km) * float(radius_km))

def make_service_mask(mesh_x, mesh_y, radius_km=None):
    if radius_km is None:
        return np.ones_like(mesh_x, dtype=bool)
    return (mesh_x**2 + mesh_y**2) <= (radius_km**2)

def safe_makedirs(path: str):
    if path and (not os.path.exists(path)):
        os.makedirs(path, exist_ok=True)

def append_run_log(out_base_dir, row: dict):
    if (not OUT_ENABLE) or (not OUT_CSV_RUNLOG):
        return
    safe_makedirs(out_base_dir)
    path = os.path.join(out_base_dir, RUNLOG_FILENAME)

    write_header = not os.path.exists(path)
    keys = list(row.keys())

    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if write_header:
            w.writeheader()
        w.writerow(row)

def save_npz_maps(outdir, tag, output_time, **arrays):
    if (not OUT_ENABLE) or (not OUT_SAVE_NUMPY) or (outdir is None):
        return
    path = os.path.join(outdir, f"maps_{tag}_{output_time}.npz")
    np.savez_compressed(path, **arrays)
    _p(f"[npz] saved: {path}")


# ===================== 3) Mesh =====================
POINT_X = np.linspace(-MESH_MAX_KM, MESH_MAX_KM, NUM_SAMPLES)
POINT_Y = np.linspace(-MESH_MAX_KM, MESH_MAX_KM, NUM_SAMPLES)
mesh_x, mesh_y = np.meshgrid(POINT_X, POINT_Y)


# ===================== 4) Load CSV =====================
runtime = time.time()

try:
    circle = pd.read_csv(INPUT_BEAM_CSV, encoding="utf-8")
except UnicodeDecodeError:
    circle = pd.read_csv(INPUT_BEAM_CSV, encoding="shift-jis")

required_cols = {"latitude", "longitude", "beam_radius", "color", "user"}
if REQUIRE_SAT_RADIUS:
    required_cols = required_cols | {"sat_radius"}

if not required_cols.issubset(set(circle.columns)):
    raise RuntimeError(
        f"入力ビームCSVに必須列が不足: 必要{sorted(required_cols)} / 現在{list(circle.columns)}"
    )

beam_center = [[circle["latitude"][i], circle["longitude"][i]] for i in range(len(circle))]
num_of_beam = len(beam_center)

# power & bandwidth (equal split)
power_W = np.full(num_of_beam, TOTAL_POWER / num_of_beam, dtype=float)
bandwidth_Hz = np.full(REPEATED_BEAM, TOTAL_BANDWIDTH / REPEATED_BEAM, dtype=float)

# output dir
output_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
output_path = os.path.join(OUTPUT_DIR_BASE, f"Result_{output_time}")
if OUT_ENABLE and OUT_MAKE_DIR:
    safe_makedirs(output_path)
else:
    output_path = None

# sat_radius
if "sat_radius" in circle.columns:
    sat_radius_m = circle["sat_radius"].astype(float).to_numpy()
else:
    if REQUIRE_SAT_RADIUS:
        raise RuntimeError("CSVに sat_radius 列がありません。配置コードで sat_radius[m] を出力してください。")
    _p("[WARN] CSVに sat_radius が無いので SAT_RADIUS_FALLBACK_M を使用します（非推奨）。")
    sat_radius_m = np.full(num_of_beam, SAT_RADIUS_FALLBACK_M, dtype=float)


# ===================== 5) Frequency groups =====================
def build_freq_groups():
    lst = [[] for _ in range(REPEATED_BEAM)]
    for i in range(num_of_beam):
        idx = int(circle["color"][i]) - 1
        idx = max(0, min(REPEATED_BEAM - 1, idx))
        lst[idx].append(i)
    return lst

freq_beam_list = build_freq_groups()

def determ_freq(beam_idx: int) -> int:
    for f_idx in range(REPEATED_BEAM):
        if beam_idx in freq_beam_list[f_idx]:
            return f_idx
    return 0


# ===================== 6) Geometry & Antenna pattern =====================
CENTER = [HAPS_LOCATION["lat"], HAPS_LOCATION["lon"]]

def beam_gain_bessel(freq_hz, dx_km, dy_km, a_m):
    c = 299792458.0
    lam = c / freq_hz
    horiz = np.sqrt(dx_km**2 + dy_km**2)
    theta = np.arctan2(horiz, HAPS_ALT_KM)
    s = (np.pi * a_m / lam) * np.sin(theta)
    return (2.0 * scipy.special.jv(1, s) / (s + 1e-12)) ** 2

def beam_gain_m2101(dx_km, dy_km, beam_radius_km, sll_db):
    horiz = np.sqrt(dx_km**2 + dy_km**2)
    theta_deg = np.degrees(np.arctan2(horiz, HAPS_ALT_KM))
    theta_3db = np.degrees(np.arctan2(beam_radius_km, HAPS_ALT_KM))
    theta_3db = max(theta_3db, 0.01)
    A_theta = 12.0 * (theta_deg / theta_3db) ** 2
    A_theta = np.minimum(A_theta, float(sll_db))
    return 10.0 ** (-A_theta / 10.0)

def beam_gain_tr38811(freq_hz, dx_km, dy_km, a_m, max_atten_db):
    c = 299792458.0
    lam = c / freq_hz
    horiz = np.sqrt(dx_km**2 + dy_km**2)
    theta = np.arctan2(horiz, HAPS_ALT_KM)
    s = (np.pi * a_m / lam) * np.sin(theta)
    g_lin = (2.0 * scipy.special.jv(1, s) / (s + 1e-12)) ** 2
    g_db = 10.0 * np.log10(g_lin + 1e-30)
    g_db = np.maximum(g_db, -float(max_atten_db))
    return 10.0 ** (g_db / 10.0)

def build_beam_offset_km():
    bx = np.zeros(num_of_beam, dtype=float)
    by = np.zeros(num_of_beam, dtype=float)
    for i in range(num_of_beam):
        lat, lon = beam_center[i]
        dx = geodesic(CENTER, [CENTER[0], lon]).km
        dy = geodesic(CENTER, [lat, CENTER[1]]).km
        if lon < CENTER[1]:
            dx *= -1.0
        if lat < CENTER[0]:
            dy *= -1.0
        bx[i] = dx
        by[i] = dy
    return bx, by

bx_km, by_km = build_beam_offset_km()

def compute_gb_maps():
    gb_maps = []
    for b in range(num_of_beam):
        dx = bx_km[b] - mesh_x
        dy = by_km[b] - mesh_y

        if ANT_PATTERN == "BESSEL":
            gb = beam_gain_bessel(CARRIER_FREQ, dx, dy, float(sat_radius_m[b]))
        elif ANT_PATTERN == "TR38811":
            gb = beam_gain_tr38811(CARRIER_FREQ, dx, dy, float(sat_radius_m[b]), TR38811_MAX_ATTEN_DB)
        elif ANT_PATTERN == "M2101":
            r_km = float(circle["beam_radius"].iloc[b])
            gb = beam_gain_m2101(dx, dy, r_km, M2101_SLL_DB)
        else:
            raise ValueError(f"Unknown ANT_PATTERN: {ANT_PATTERN}")

        gb_maps.append(gb.astype(np.float32))
    return gb_maps

gb_maps = compute_gb_maps()


# ===================== 7) Link budget field =====================
def dl_field_mw():
    if USE_DISTANCE_FSPL:
        d_m = np.sqrt((HAPS_ALT_KM**2 + mesh_x**2 + mesh_y**2)) * 1000.0
        fspl = fspl_db(d_m, CARRIER_FREQ)
    else:
        d0_m = (HAPS_ALT_KM * 1000.0)
        fspl = fspl_db(d0_m, CARRIER_FREQ)

    dl_db = TX_GAIN_DB + RX_GAIN_DB - fspl - MISC_LOSS_DB
    return from_dB(dl_db).astype(np.float32)

dl_mw_xy = dl_field_mw()


# ===================== 8) Throughput mapping =====================
def sinr_db_to_cqi_and_eff(sinr_db):
    thr = np.array([t[2] for t in CQI_TABLE], dtype=float)
    eff = np.array([t[1] for t in CQI_TABLE], dtype=float)
    idx = np.sum(sinr_db[..., None] >= thr[None, None, :], axis=-1) - 1
    idx = np.clip(idx, -1, len(eff) - 1)
    out = np.where(idx >= 0, eff[idx], 0.0)
    return out.astype(np.float32)

def sinr_to_eff_shannon(sinr_lin):
    gap = from_dB(SHANNON_GAP_DB)
    eta = np.log2(1.0 + sinr_lin / gap)
    eta = np.clip(eta, ETA_MIN, ETA_MAX)
    return eta.astype(np.float32)

def sinr_to_eff(sinr_lin):
    mode = THROUGHPUT_MODE.upper()
    if mode == "CQI":
        sinr_db = to_dB(sinr_lin)
        return sinr_db_to_cqi_and_eff(sinr_db)
    if mode == "SHANNON":
        return sinr_to_eff_shannon(sinr_lin)
    raise ValueError(f"Unknown THROUGHPUT_MODE: {THROUGHPUT_MODE}")


# ===================== 9) Fast best-server evaluation =====================
def precompute_rx_maps_by_beam():
    rx_map = [None] * num_of_beam
    total_rx_by_color = [np.zeros_like(mesh_x, dtype=np.float32) for _ in range(REPEATED_BEAM)]
    for b in range(num_of_beam):
        f = determ_freq(b)
        rx = (power_W[b] * 1000.0) * gb_maps[b] * dl_mw_xy
        rx_map[b] = rx
        total_rx_by_color[f] += rx
    return rx_map, total_rx_by_color

def best_server_eval(mask):
    rx_map, total_rx_by_color = precompute_rx_maps_by_beam()

    best_bps = np.full_like(mesh_x, np.nan, dtype=np.float32)
    best_eff = np.full_like(mesh_x, np.nan, dtype=np.float32)
    best_sinr_db = np.full_like(mesh_x, np.nan, dtype=np.float32)
    best_beam_idx = np.full_like(mesh_x, -1, dtype=np.int32)

    best_bps[mask] = 0.0
    best_eff[mask] = 0.0
    best_sinr_db[mask] = -1e9

    for b in range(num_of_beam):
        f = determ_freq(b)
        carrier = rx_map[b]
        interference = total_rx_by_color[f] - carrier
        noise = float(calc_noise_mw(bandwidth_Hz[f]))

        sinr = carrier / (noise + interference + 1e-30)
        eff = sinr_to_eff(sinr)
        bps = eff * float(bandwidth_Hz[f])

        bps_masked = np.where(mask, bps, -1.0)
        update = bps_masked > best_bps

        best_bps[update] = bps[update]
        best_eff[update] = eff[update]
        best_sinr_db[update] = to_dB(sinr)[update]
        best_beam_idx[update] = b

    return best_beam_idx, best_sinr_db, best_eff, best_bps


# ===================== 9.2) Realistic capacity (beam-averaged) =====================
def calc_system_capacity_realistic(num_beam, best_beam_idx, best_eff, bandwidth_array, freq_func, eval_mask):
    total_cap_bps = 0.0
    valid_beam_count = 0

    for b in range(num_beam):
        m = (best_beam_idx == b) & (best_eff > 0.0) & eval_mask
        if np.any(m):
            avg_eff = float(np.mean(best_eff[m]))
            f_idx = int(freq_func(b))
            bw = float(bandwidth_array[f_idx])
            total_cap_bps += (avg_eff * bw)
            valid_beam_count += 1

    return float(total_cap_bps), int(valid_beam_count)


# ===================== 9.5) Coverage / Quality metrics =====================
def compute_quality_metrics(eval_mask, best_beam_idx, best_sinr_db, best_eff, best_bps):
    inside = eval_mask & (~np.isnan(best_bps))
    total_pts = int(np.sum(inside))
    if total_pts == 0:
        return {
            "total_points": 0,
            "connected_points": 0,
            "low_quality_points": 0,
            "unconnected_points": 0,
            "coverage_rate": np.nan,
            "low_quality_rate_all": np.nan,
            "low_quality_rate_connected": np.nan,
            "unconnected_rate": np.nan,
        }

    connected = inside & (best_beam_idx >= 0) & (best_eff > 0.0) & (best_sinr_db >= CONNECT_SINR_MIN_DB)
    unconnected = inside & (~connected)
    low_quality = connected & (best_sinr_db < LOW_QUALITY_SINR_DB)

    n_conn = int(np.sum(connected))
    n_unconn = int(np.sum(unconnected))
    n_low = int(np.sum(low_quality))

    coverage_rate = n_conn / total_pts
    unconnected_rate = n_unconn / total_pts
    low_quality_rate_all = n_low / total_pts
    low_quality_rate_connected = (n_low / n_conn) if n_conn > 0 else np.nan

    return {
        "total_points": total_pts,
        "connected_points": n_conn,
        "low_quality_points": n_low,
        "unconnected_points": n_unconn,
        "coverage_rate": float(coverage_rate),
        "low_quality_rate_all": float(low_quality_rate_all),
        "low_quality_rate_connected": float(low_quality_rate_connected) if not np.isnan(low_quality_rate_connected) else np.nan,
        "unconnected_rate": float(unconnected_rate),
    }


# ===================== 10) Summary utilities and exports =====================
def summarize_map_stats(arr, eval_mask):
    v = arr[eval_mask & (~np.isnan(arr))]
    if v.size == 0:
        return {"mean": np.nan, "median": np.nan, "p5": np.nan, "p95": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "p5": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }

def export_best_server_summary_csv(outdir, tag, best_sinr_db, best_eff, best_bps, eval_mask):
    if (not OUT_ENABLE) or (not OUT_CSV_BEST_SERVER_SUM) or (outdir is None):
        return
    sinr_stats = summarize_map_stats(best_sinr_db, eval_mask)
    eff_stats  = summarize_map_stats(best_eff, eval_mask)
    bps_stats  = summarize_map_stats(best_bps, eval_mask)

    path = os.path.join(outdir, f"best_server_summary_{tag}_{output_time}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric","mean","median","p5","p95","min","max"])
        w.writerow(["best_SINR_dB", sinr_stats["mean"], sinr_stats["median"], sinr_stats["p5"], sinr_stats["p95"], sinr_stats["min"], sinr_stats["max"]])
        w.writerow(["best_eff_bps_per_Hz", eff_stats["mean"], eff_stats["median"], eff_stats["p5"], eff_stats["p95"], eff_stats["min"], eff_stats["max"]])
        w.writerow(["best_bps", bps_stats["mean"], bps_stats["median"], bps_stats["p5"], bps_stats["p95"], bps_stats["min"], bps_stats["max"]])
    _p(f"[best-server] summary saved: {path}")

def export_grid_quality_metrics_csv(outdir, tag, metrics: dict, eval_name: str):
    if (not OUT_ENABLE) or (not OUT_CSV_GRID_METRICS) or (outdir is None):
        return
    path = os.path.join(outdir, f"grid_quality_metrics_{tag}_{eval_name}_{output_time}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "value"])
        w.writerow(["SERVICE_RADIUS_KM", SERVICE_RADIUS_KM])
        w.writerow(["NUM_SAMPLES", NUM_SAMPLES])
        w.writerow(["MESH_MAX_KM", MESH_MAX_KM])
        w.writerow(["CONNECT_SINR_MIN_DB", CONNECT_SINR_MIN_DB])
        w.writerow(["LOW_QUALITY_SINR_DB", LOW_QUALITY_SINR_DB])
        for k, v in metrics.items():
            w.writerow([k, v])
    _p(f"[metrics] saved: {path}")

def export_result_summary(iter_tag, total_bitrate_proxy, outdir, eval_name: str):
    if (not OUT_ENABLE) or (not OUT_CSV_RESULT_SUMMARY) or (outdir is None):
        return
    path = os.path.join(outdir, f"result_{num_of_beam}beam_{eval_name}_{output_time}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iter","Beams","Reuse","TotalPower_W","TotalBW_Hz","ThroughputMode","AntPattern","UseDistFSPL","total_bitrate_proxy_bps"])
        w.writerow([iter_tag, num_of_beam, REPEATED_BEAM, TOTAL_POWER, TOTAL_BANDWIDTH, THROUGHPUT_MODE, ANT_PATTERN, USE_DISTANCE_FSPL, total_bitrate_proxy])
    _p(f"[result] saved: {path}")


# ===================== 11) Plots =====================
def plot_maps(best_bps, best_sinr_db, tag, outdir, eval_mask):
    if (not OUT_ENABLE) or (not OUT_PLOT_MAP):
        return

    bps_plot = np.where(eval_mask, best_bps, np.nan)
    sinr_plot = np.where(eval_mask, best_sinr_db, np.nan)

    plt.figure(figsize=(10, 8))
    plt.pcolormesh(mesh_x, mesh_y, bps_plot, shading="auto")
    plt.colorbar(label="Best-server throughput [bps] (per grid point)")
    plt.title(f"HAPS Throughput Map (Best-server) [{tag}]")
    plt.xlabel("X [km]")
    plt.ylabel("Y [km]")
    if OUT_SAVE_MAP and (outdir is not None):
        plt.savefig(os.path.join(outdir, f"throughput_map_{tag}_{output_time}.png"), dpi=160, bbox_inches="tight")

    plt.figure(figsize=(10, 8))
    plt.pcolormesh(mesh_x, mesh_y, sinr_plot, shading="auto")
    plt.colorbar(label="Best-server SINR [dB]")
    plt.title(f"HAPS SINR Map (Best-server) [{tag}]")
    plt.xlabel("X [km]")
    plt.ylabel("Y [km]")
    if OUT_SAVE_MAP and (outdir is not None):
        plt.savefig(os.path.join(outdir, f"sinr_map_{tag}_{output_time}.png"), dpi=160, bbox_inches="tight")


# ===================== 12) main =====================
def run_one_eval(eval_name: str, eval_mask: np.ndarray):
    best_idx, best_sinr_db, best_eff, best_bps = best_server_eval(mask=eval_mask)

    proxy_sum_bps = float(np.nansum(np.where(eval_mask, best_bps, np.nan)))

    real_cap_bps, active_beams = calc_system_capacity_realistic(
        num_of_beam, best_idx, best_eff, bandwidth_Hz, determ_freq, eval_mask=eval_mask
    )
    voice_users = real_cap_bps / 64000.0

    metrics = compute_quality_metrics(eval_mask, best_idx, best_sinr_db, best_eff, best_bps)

    tag = f"{THROUGHPUT_MODE}_{ANT_PATTERN}_{eval_name}"
    export_result_summary("HAPS_best_server", proxy_sum_bps, output_path, eval_name=eval_name)
    export_best_server_summary_csv(output_path, tag, best_sinr_db, best_eff, best_bps, eval_mask=eval_mask)
    export_grid_quality_metrics_csv(output_path, tag, metrics, eval_name=eval_name)

    save_npz_maps(
        output_path, tag, output_time,
        best_idx=best_idx, best_sinr_db=best_sinr_db, best_eff=best_eff, best_bps=best_bps,
        mask=eval_mask.astype(np.uint8)
    )

    plot_maps(best_bps, best_sinr_db, tag=tag, outdir=output_path, eval_mask=eval_mask)

    return {
        "proxy_sum_bps": proxy_sum_bps,
        "real_cap_bps": real_cap_bps,
        "active_beams": active_beams,
        "voice_users_64kbps": float(voice_users),
        "coverage_rate": metrics.get("coverage_rate", np.nan),
        "low_quality_rate_all": metrics.get("low_quality_rate_all", np.nan),
        "low_quality_rate_connected": metrics.get("low_quality_rate_connected", np.nan),
        "unconnected_rate": metrics.get("unconnected_rate", np.nan),
        "metrics": metrics,
    }

def main():
    _p(f"HAPS sim start @ {HAPS_LOCATION}, Alt={HAPS_ALT_KM} km, Freq={CARRIER_FREQ/1e9:.2f} GHz")
    _p(f"Beams={num_of_beam}, Reuse={REPEATED_BEAM}, TotalBW={TOTAL_BANDWIDTH/1e6:.1f} MHz, TotalP={TOTAL_POWER} W")
    _p(f"THROUGHPUT_MODE={THROUGHPUT_MODE}, ANT_PATTERN={ANT_PATTERN}, USE_DISTANCE_FSPL={USE_DISTANCE_FSPL}")
    _p(f"sat_radius[m] min/max = {float(np.min(sat_radius_m)):.3f} / {float(np.max(sat_radius_m)):.3f}")
    _p(f"CONNECT_SINR_MIN_DB={CONNECT_SINR_MIN_DB} dB, LOW_QUALITY_SINR_DB={LOW_QUALITY_SINR_DB} dB")

    # masks
    all_mask = make_service_mask(mesh_x, mesh_y, SERVICE_RADIUS_KM)

    if DISASTER_ENABLE:
        disaster_mask = make_circle_mask(
            mesh_x, mesh_y,
            DISASTER_CENTER_X_KM, DISASTER_CENTER_Y_KM,
            DISASTER_RADIUS_KM
        )
    else:
        disaster_mask = None

    # run evaluations
    results = {}

    if DISASTER_ENABLE:
        results["disaster"] = run_one_eval("disaster", disaster_mask)
        _p(f"[disaster metrics] {results['disaster']['metrics']}")
        if DISASTER_EXPORT_ALL_TOO:
            results["all"] = run_one_eval("all", all_mask)
            _p(f"[all metrics] {results['all']['metrics']}")
    else:
        results["all"] = run_one_eval("all", all_mask)
        _p(f"[all metrics] {results['all']['metrics']}")

    # append run log (one line)
    row = {
        "time": output_time,
        "beam_csv": os.path.basename(INPUT_BEAM_CSV),
        "output_dir": (output_path if output_path is not None else ""),
        "num_beam": num_of_beam,
        "reuse": REPEATED_BEAM,
        "total_bw_hz": TOTAL_BANDWIDTH,
        "total_power_w": TOTAL_POWER,
        "service_radius_km": SERVICE_RADIUS_KM,
        "mesh_max_km": MESH_MAX_KM,
        "num_samples": NUM_SAMPLES,
        "ant_pattern": ANT_PATTERN,
        "m2101_sll_db": M2101_SLL_DB,
        "tr38811_max_atten_db": TR38811_MAX_ATTEN_DB,
        "throughput_mode": THROUGHPUT_MODE,
        "use_distance_fspl": USE_DISTANCE_FSPL,
        "connect_sinr_min_db": CONNECT_SINR_MIN_DB,
        "low_quality_sinr_db": LOW_QUALITY_SINR_DB,
        "disaster_center_x_km": (DISASTER_CENTER_X_KM if DISASTER_ENABLE else ""),
        "disaster_center_y_km": (DISASTER_CENTER_Y_KM if DISASTER_ENABLE else ""),
        "disaster_radius_km": (DISASTER_RADIUS_KM if DISASTER_ENABLE else ""),
        "runtime_s": (time.time() - runtime),
    }

    if "all" in results:
        row.update({
            "proxy_sum_bps_all": results["all"]["proxy_sum_bps"],
            "real_cap_bps_all": results["all"]["real_cap_bps"],
            "active_beams_all": results["all"]["active_beams"],
            "voice_users_64kbps_all": results["all"]["voice_users_64kbps"],
            "coverage_rate_all": results["all"]["coverage_rate"],
            "low_quality_rate_all": results["all"]["low_quality_rate_all"],
            "low_quality_rate_connected_all": results["all"]["low_quality_rate_connected"],
            "unconnected_rate_all": results["all"]["unconnected_rate"],
        })

    if "disaster" in results:
        row.update({
            "proxy_sum_bps_disaster": results["disaster"]["proxy_sum_bps"],
            "real_cap_bps_disaster": results["disaster"]["real_cap_bps"],
            "active_beams_disaster": results["disaster"]["active_beams"],
            "voice_users_64kbps_disaster": results["disaster"]["voice_users_64kbps"],
            "coverage_rate_disaster": results["disaster"]["coverage_rate"],
            "low_quality_rate_disaster": results["disaster"]["low_quality_rate_all"],
            "low_quality_rate_connected_disaster": results["disaster"]["low_quality_rate_connected"],
            "unconnected_rate_disaster": results["disaster"]["unconnected_rate"],
        })

    append_run_log(OUTPUT_DIR_BASE, row)

    if OUT_SHOW_PLOT:
        plt.show()

    _p(f"処理時間 = {time.time() - runtime:.2f} s")
    _p(f"output_dir = {output_path}")


if __name__ == "__main__":
    main()
