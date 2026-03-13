# -*- coding: utf-8 -*-
"""
開口半径 a は固定のまま、角度をスケーリングしてビーム幅だけ変える
HAPS downlink simulation (Evaluation: Capacity & SINR Quality)
- 軸1: ベストエフォート (System Capacity)
- 軸2: 品質・公平性 (SINR 5%-tile, SINR StdDev)
"""

# ===================== imports =====================
import os
import csv
import time
import datetime
import numpy as np
import pandas as pd
import scipy.special
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from global_land_mask import globe

# ===================== 1) Global Config =====================
# ---- HAPS parameters ----
HAPS_LOCATION = {'lat': 26.4816, 'lon': 127.9755}
HAPS_ALT_KM   = 20.0
CARRIER_FREQ  = 2.0e9
REPEATED_BEAM = 3     # マルチビームなら3, シングルなら1
SERVICE_RADIUS_KM = 50.0

TOTAL_BANDWIDTH = 20e6
TOTAL_POWER     = 20.0 

# ---- Antenna parameters (Fixed Gain) ----
TX_GAIN_DB = 24.0     
R_REF_KM = 2.7
POINTING_ERROR_DEG  = 0.0 

RX_GAIN_DB      = 0.0
MISC_LOSS_DB    = 0.0
ANT_PATTERN = "TR38811"
TR38811_MAX_ATTEN_DB = 30.0
USE_DISTANCE_FSPL = True 

# ---- Throughput mapping ----
THROUGHPUT_MODE = "SHANNON"
SHANNON_GAP_DB = 0.0
ETA_MAX = 6.0
ETA_MIN = 0.0
CONNECT_SINR_MIN_DB = -6.7
LOW_QUALITY_SINR_DB = 0.0

# ---- Disaster-area evaluation ----
DISASTER_ENABLE = False
DISASTER_MODE = "CIRCLE"
DISASTER_CENTER_X_KM = 21.0
DISASTER_CENTER_Y_KM = 25.0
DISASTER_RADIUS_KM = 18.0

# ---- IO Settings ----
OUTPUT_DIR_BASE = r"Result\BatchRun\4\p80"
OUT_ENABLE = True
OUT_MAKE_DIR = True
OUT_PRINT = True
OUT_CSV_RUNLOG = True
RUNLOG_FILENAME = "run_log_sinr_eval.csv"

# ---- Mesh (Global) ----
MESH_MAX_KM = 100.0
NUM_SAMPLES = 401
POINT_X = np.linspace(-MESH_MAX_KM, MESH_MAX_KM, NUM_SAMPLES)
POINT_Y = np.linspace(-MESH_MAX_KM, MESH_MAX_KM, NUM_SAMPLES)
mesh_x, mesh_y = np.meshgrid(POINT_X, POINT_Y)


# ===================== 2) Utility Functions =====================
def _p(msg: str):
    if OUT_PRINT: print(msg)

def to_dB(x): return 10.0 * np.log10(x + 1e-30)
def from_dB(dB_val): return 10.0 ** (dB_val / 10.0)

def fspl_db(d_m, f_hz):
    c = 299792458.0
    lam = c / f_hz
    return 20.0 * np.log10(4.0 * np.pi * d_m / lam + 1e-30)

def calc_noise_mw(bandwidth_Hz):
    K_BOLTZ = 1.38064852e-23
    UE_NF_DB = 3.0
    T_SYS = 290.0 * (10 ** (UE_NF_DB / 10.0))
    return K_BOLTZ * T_SYS * bandwidth_Hz * 1000.0

def make_service_mask(mesh_x, mesh_y, radius_km=None):
    if radius_km is None: return np.ones_like(mesh_x, dtype=bool)
    return (mesh_x**2 + mesh_y**2) <= (radius_km**2)

def safe_makedirs(path: str):
    if path and (not os.path.exists(path)): os.makedirs(path, exist_ok=True)

def append_run_log(out_base_dir, row: dict):
    if (not OUT_ENABLE) or (not OUT_CSV_RUNLOG): return
    safe_makedirs(out_base_dir)
    path = os.path.join(out_base_dir, RUNLOG_FILENAME)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)

def make_land_mask(mesh_x, mesh_y, center_lat, center_lon):
    lat_deg_len_km = 110.574
    lon_deg_len_km = 111.320 * np.cos(np.deg2rad(center_lat))
    lat_grid = center_lat + (mesh_y / lat_deg_len_km)
    lon_grid = center_lon + (mesh_x / lon_deg_len_km)
    return globe.is_land(lat_grid, lon_grid)

def calc_aperture_radius_for_target_width(target_radius_km, freq_hz):
    c = 299792458.0
    lam = c / freq_hz
    theta_rad = np.arctan(target_radius_km / HAPS_ALT_KM)
    sin_theta = np.sin(theta_rad)
    x_half = 1.61633
    a_m = (x_half * lam) / (np.pi * sin_theta)
    return a_m

def calc_pointing_loss_db(beam_radius_km, error_deg=POINTING_ERROR_DEG):
    if error_deg <= 0.0: return 0.0
    theta_half_rad = np.arctan(beam_radius_km / HAPS_ALT_KM)
    theta_3db_deg = np.degrees(theta_half_rad) * 2.0
    loss_db = 12.0 * (error_deg / theta_3db_deg) ** 2
    return loss_db

# ===================== 3) Core Simulation Logic =====================
def beam_gain_tr38811_normalized(freq_hz, dx_km, dy_km, a_m, max_atten_db):
    c = 299792458.0
    lam = c / freq_hz
    horiz = np.sqrt(dx_km**2 + dy_km**2)
    theta = np.arctan2(horiz, HAPS_ALT_KM)
    s = (np.pi * a_m / lam) * np.sin(theta)
    g_lin = (2.0 * scipy.special.jv(1, s) / (s + 1e-12)) ** 2
    g_db = 10.0 * np.log10(g_lin + 1e-30)
    g_db = np.maximum(g_db, -float(max_atten_db))
    return 10.0 ** (g_db / 10.0)

def theta_from_radius_km(beam_radius_km):
    # 「この半径がビームの代表幅（半値幅相当）」という設計で角度へ変換
    return np.arctan(beam_radius_km / HAPS_ALT_KM)  # rad

def beam_gain_tr38811_normalized_scaled(freq_hz, dx_km, dy_km, a_m_fixed, max_atten_db, beam_radius_km, r_ref_km=R_REF_KM):
    c = 299792458.0
    lam = c / freq_hz

    horiz = np.sqrt(dx_km**2 + dy_km**2)
    theta = np.arctan2(horiz, HAPS_ALT_KM)  # 実角度

    # 角度スケーリング（幅だけ調整）
    th_ref = theta_from_radius_km(r_ref_km)
    th_tgt = theta_from_radius_km(beam_radius_km)
    scale = th_ref / (th_tgt + 1e-12)
    theta_scaled = theta * scale

    s = (np.pi * a_m_fixed / lam) * np.sin(theta_scaled)
    g_lin = (2.0 * scipy.special.jv(1, s) / (s + 1e-12)) ** 2
    g_db = 10.0 * np.log10(g_lin + 1e-30)
    g_db = np.maximum(g_db, -float(max_atten_db))
    return 10.0 ** (g_db / 10.0)


def run_simulation(radius_label, csv_path, storage_sinr):
    start_time = time.time()
    
    if not os.path.exists(csv_path):
        _p(f"[SKIP] File not found: {csv_path}")
        return

    _p(f"\n>>> Start Sim: R={radius_label}km [Disaster={DISASTER_ENABLE}]")
    
    try:
        circle = pd.read_csv(csv_path, encoding="utf-8")
    except:
        circle = pd.read_csv(csv_path, encoding="shift-jis")

    beam_center = [[circle["latitude"][i], circle["longitude"][i]] for i in range(len(circle))]
    num_of_beam = len(beam_center)
    
    power_W = np.full(num_of_beam, TOTAL_POWER / num_of_beam, dtype=float)
    bandwidth_Hz = np.full(REPEATED_BEAM, TOTAL_BANDWIDTH / REPEATED_BEAM, dtype=float)

    # perture は固定（参照半径から一度だけ決める）
    a_m_fixed = calc_aperture_radius_for_target_width(R_REF_KM, CARRIER_FREQ)

    p_loss_db = calc_pointing_loss_db(radius_label, POINTING_ERROR_DEG)

    freq_groups = [[] for _ in range(REPEATED_BEAM)]
    for i in range(num_of_beam):
        idx = int(circle["color"][i]) - 1 if "color" in circle.columns else 0
        idx = max(0, min(REPEATED_BEAM - 1, idx))
        freq_groups[idx].append(i)
    
    def get_freq(b_idx):
        for f in range(REPEATED_BEAM):
            if b_idx in freq_groups[f]: return f
        return 0

    bx_km = np.zeros(num_of_beam)
    by_km = np.zeros(num_of_beam)
    CENTER = [HAPS_LOCATION["lat"], HAPS_LOCATION["lon"]]
    for i in range(num_of_beam):
        lat, lon = beam_center[i]
        dx = geodesic(CENTER, [CENTER[0], lon]).km
        dy = geodesic(CENTER, [lat, CENTER[1]]).km
        if lon < CENTER[1]: dx *= -1.0
        if lat < CENTER[0]: dy *= -1.0
        bx_km[i], by_km[i] = dx, dy

    gb_maps = []
    for b in range(num_of_beam):
        dx = bx_km[b] - mesh_x
        dy = by_km[b] - mesh_y
        #gb = beam_gain_tr38811_normalized(CARRIER_FREQ, dx, dy, float(sat_radius_m[b]), TR38811_MAX_ATTEN_DB)
        gb = beam_gain_tr38811_normalized_scaled(CARRIER_FREQ, dx, dy, a_m_fixed, TR38811_MAX_ATTEN_DB, beam_radius_km=float(radius_label), r_ref_km=R_REF_KM
)

        gb_maps.append(gb.astype(np.float32))

    if USE_DISTANCE_FSPL:
        d_m = np.sqrt((HAPS_ALT_KM**2 + mesh_x**2 + mesh_y**2)) * 1000.0
        fspl = fspl_db(d_m, CARRIER_FREQ)
    else:
        d0_m = (HAPS_ALT_KM * 1000.0)
        fspl = fspl_db(d0_m, CARRIER_FREQ)
        
    tx_gain_linear_fixed = from_dB(TX_GAIN_DB)
    dl_mw_factor = tx_gain_linear_fixed * from_dB(RX_GAIN_DB - fspl - MISC_LOSS_DB - p_loss_db).astype(np.float32)

    # --- Best Server Calculation ---
    rx_map = [None] * num_of_beam
    total_rx_by_color = [np.zeros_like(mesh_x, dtype=np.float32) for _ in range(REPEATED_BEAM)]
    
    for b in range(num_of_beam):
        f = get_freq(b)
        rx = (power_W[b] * 1000.0) * dl_mw_factor * gb_maps[b]
        rx_map[b] = rx
        total_rx_by_color[f] += rx
        
    service_mask = make_service_mask(mesh_x, mesh_y, SERVICE_RADIUS_KM)
    
    disaster_mask = np.zeros_like(mesh_x, dtype=bool)
    if DISASTER_ENABLE and DISASTER_MODE == "CIRCLE":
        dist_sq = (mesh_x - DISASTER_CENTER_X_KM)**2 + (mesh_y - DISASTER_CENTER_Y_KM)**2
        disaster_mask = dist_sq <= (DISASTER_RADIUS_KM**2)

    best_bps = np.full_like(mesh_x, np.nan, dtype=np.float32)
    best_eff = np.full_like(mesh_x, np.nan, dtype=np.float32)
    best_sinr_db = np.full_like(mesh_x, np.nan, dtype=np.float32)
    best_beam_idx = np.full_like(mesh_x, -1, dtype=np.int32)
    
    best_bps[service_mask] = 0.0
    best_eff[service_mask] = 0.0
    best_sinr_db[service_mask] = -1e9

    for b in range(num_of_beam):
        f = get_freq(b)
        carrier = rx_map[b]
        interference = total_rx_by_color[f] - carrier
        noise = float(calc_noise_mw(bandwidth_Hz[f]))
        sinr = carrier / (noise + interference + 1e-30)
        
        gap = from_dB(SHANNON_GAP_DB)
        eff = np.log2(1.0 + sinr / gap)
        eff = np.clip(eff, 0.0, ETA_MAX)
        bps = eff * float(bandwidth_Hz[f])
        
        bps_masked = np.where(service_mask, bps, -1.0)
        update = bps_masked > best_bps
        best_bps[update] = bps[update]
        best_eff[update] = eff[update]
        best_sinr_db[update] = to_dB(sinr)[update]
        best_beam_idx[update] = b

    # --- Metrics & Stats ---
    is_land = make_land_mask(mesh_x, mesh_y, HAPS_LOCATION["lat"], HAPS_LOCATION["lon"])
    
    if DISASTER_ENABLE:
        target_area = disaster_mask & is_land
    else:
        target_area = service_mask & is_land

    inside = target_area & (~np.isnan(best_bps))
    total_pts = int(np.sum(inside))
    
    # 統計変数の初期化
    sinr_mean = np.nan
    sinr_var  = np.nan
    sinr_std  = np.nan
    sinr_p5   = np.nan  # 5%-tile
    sinr_p50  = np.nan  # Median
    
    if total_pts > 0:
        connected = inside & (best_beam_idx >= 0) & (best_sinr_db >= CONNECT_SINR_MIN_DB)
        low_quality = connected & (best_sinr_db < LOW_QUALITY_SINR_DB)
        
        coverage_rate = np.sum(connected) / total_pts
        low_quality_rate_all = np.sum(low_quality) / total_pts
        
        # SINR Stats (Distribution Analysis)
        # 極端な低値(-100dB以下)は計算から除外
        sinr_vals = best_sinr_db[inside]
        sinr_vals = sinr_vals[sinr_vals > -100]
        
        if len(sinr_vals) > 0:
            sinr_mean = np.mean(sinr_vals)
            sinr_var  = np.var(sinr_vals)
            sinr_std  = np.sqrt(sinr_var)
            sinr_p5   = np.percentile(sinr_vals, 5)   # ★追加: 帯域保証の指標 (5%)
            sinr_p50  = np.percentile(sinr_vals, 50)  # Median
            
            # CDF保存用
            storage_sinr[radius_label] = sinr_vals
    else:
        coverage_rate = np.nan
        low_quality_rate_all = np.nan

    # System Capacity (Best Effort)
    real_cap_bps = 0.0
    for b in range(num_of_beam):
        area_mask = (best_beam_idx == b) & target_area 
        if np.any(area_mask):
            avg_eff = float(np.mean(best_eff[area_mask]))
            f_idx = get_freq(b)
            real_cap_bps += (avg_eff * bandwidth_Hz[f_idx])

    # Log
    output_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    row = {
        "radius_km": radius_label,
        "time": output_time,
        "csv": os.path.basename(csv_path),
        "disaster_mode": DISASTER_ENABLE,
        "real_cap_mbps": real_cap_bps / 1e6, # 軸1: ベストエフォート
        "sinr_p5_db": sinr_p5,               # 軸2: 帯域保証(Cell Edge)
        "sinr_std_db": sinr_std,             # 軸2: 公平性(分散)
        "sinr_mean_db": sinr_mean,
        "low_quality_rate": low_quality_rate_all,
        "total_pts": total_pts,
    }
    append_run_log(OUTPUT_DIR_BASE, row)
    
    _p(f"   -> Cap={real_cap_bps/1e6:.1f} Mbps, SINR P5={sinr_p5:.2f}dB, Std={sinr_std:.2f}")

# ===================== 4) Plotting Functions =====================
def plot_and_save_cdf(storage_dict, xlabel, filename_suffix, xlim=None):
    plt.figure(figsize=(10, 7))
    if len(storage_dict) == 0:
        plt.text(0.5, 0.5, "No Data", ha='center')
    else:
        colors = plt.cm.jet(np.linspace(0, 1, len(storage_dict)))
        plotted_any = False
        for (label, data), color in zip(storage_dict.items(), colors):
            if len(data) == 0: continue
            sorted_data = np.sort(data)
            yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
            plt.plot(sorted_data, yvals, label=f"R={label}km", color=color, linewidth=2)
            plotted_any = True
        if plotted_any: plt.legend()

    plt.xlabel(xlabel)
    plt.ylabel("CDF (Probability)")
    plt.title(f"CDF: {xlabel} (Disaster Area: {DISASTER_ENABLE})")
    plt.grid(True, linestyle='--', alpha=0.7)
    if xlim: plt.xlim(xlim)
    plt.savefig(os.path.join(OUTPUT_DIR_BASE, f"CDF_{filename_suffix}.png"))
    plt.close()

def plot_and_save_pdf(storage_dict, xlabel, filename_suffix, xlim=None):
    plt.figure(figsize=(10, 7))
    if len(storage_dict) == 0:
        plt.text(0.5, 0.5, "No Data", ha='center')
    else:
        colors = plt.cm.jet(np.linspace(0, 1, len(storage_dict)))
        plotted_any = False
        for (label, data), color in zip(storage_dict.items(), colors):
            if len(data) < 2: continue
            try:
                density = gaussian_kde(data)
                if xlim:
                    xs = np.linspace(xlim[0], xlim[1], 200)
                else:
                    xs = np.linspace(min(data), max(data), 200)
                plt.plot(xs, density(xs), label=f"R={label}km", color=color, linewidth=2)
                plt.fill_between(xs, density(xs), color=color, alpha=0.1)
                plotted_any = True
            except:
                pass
        if plotted_any: plt.legend()

    plt.xlabel(xlabel)
    plt.ylabel("Probability Density")
    plt.title(f"PDF: {xlabel} (Disaster Area: {DISASTER_ENABLE})")
    plt.grid(True, linestyle='--', alpha=0.7)
    if xlim: plt.xlim(xlim)
    plt.savefig(os.path.join(OUTPUT_DIR_BASE, f"PDF_{filename_suffix}.png"))
    plt.close()

# ===================== 5) Batch Execution =====================
if __name__ == "__main__":
    #_p("=== Disaster Simulation: Yambaru Pattern A ===")
    
    storage_sinr = {}
    base_path = r"database\beamlist_p80"
    scenarios = [
        (1.0,  os.path.join(base_path, "HAPS_beamlist_r1km_p0.8.csv")),
        (2.0,  os.path.join(base_path, "HAPS_beamlist_r2km_p0.8.csv")),
        (3.0,  os.path.join(base_path, "HAPS_beamlist_r3km_p0.8.csv")),
        (4.0,  os.path.join(base_path, "HAPS_beamlist_r4km_p0.8.csv")),
        (5.0,  os.path.join(base_path, "HAPS_beamlist_r5km_p0.8.csv")),
        (6.0,  os.path.join(base_path, "HAPS_beamlist_r6km_p0.8.csv")),
        (7.0,  os.path.join(base_path, "HAPS_beamlist_r7km_p0.8.csv")),
        (8.0,  os.path.join(base_path, "HAPS_beamlist_r8km_p0.8.csv")),
        (9.0,  os.path.join(base_path, "HAPS_beamlist_r9km_p0.8.csv")),
        (10.0, os.path.join(base_path, "HAPS_beamlist_r10km_p0.8.csv")),
    ]
    # ★INPUT: 災害用ビームリスト (必要に応じて変更)
 #   input_csv = r"database\beamlist_disaster\HAPS_beamlist_r5km_p1_disaster_cx21_cy25_r18.csv"
 #   
 #   scenarios = [
 #       (5.0, input_csv),
 #   ]

    for r, f in scenarios:
        run_simulation(r, f, storage_sinr)

    if OUT_ENABLE:
        safe_makedirs(OUTPUT_DIR_BASE)
        # SINR CDF (軸2: 公平性/保証の可視化)
        plot_and_save_cdf(storage_sinr, "SINR [dB]", "SINR", xlim=(-5, 25))
        # SINR PDF (分布形状の確認)
        plot_and_save_pdf(storage_sinr, "SINR [dB]", "SINR", xlim=(-5, 25))
        
    _p("\n=== All Completed ===")