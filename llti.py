"""
llti.py  –  HRRR-based Low-Level Turbulence Index (LLTI)

Adapts scott.minnick's GFE LLTI Smart Tool to HRRR gridded data over Colorado.

Algorithm is preserved exactly from the GFE version:
  score = W_MIX   * s_mix_eff   (gated by transport wind speed)
        + W_TWSPD * s_twspd
        + W_SKY   * s_sky        (inverted – clearer sky → higher score)
        + W_DD    * s_dd         (drier air → higher score)
  LLTI = clip(score, 0, 1) * 100

HRRR variable mapping:
  GFE MixHgt    → HPBL (m → ft)                             sfc product
  GFE TransWind → HPBL-coupled thickness-weighted mean wind  sfc + prs products
                  Levels: 10m anchor + 950/925/900/875/850/
                          825/800/750/700 mb
                  Only levels with midpoint AGL < HPBL contribute.
                  Boundary layer thickness weighting (midpoint rule).
  GFE Sky       → TCDC entire atmosphere (%)                 sfc product
  GFE T         → TMP 2 m above ground (K → °F)             sfc product
  GFE Td        → DPT 2 m above ground (K → °F)             sfc product
"""

import io
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from herbie import Herbie
import xarray as xr

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Algorithm constants  (mirror the GFE tool exactly)
# ─────────────────────────────────────────────────────────────────────────────
MIX_LO,   MIX_HI   = 5_000.0, 12_000.0   # ft – mixing height thresholds
TWSPD_LO, TWSPD_HI =    20.0,     60.0   # kt – transport wind thresholds
SKY_REF              =    70.0             # % – sky cover reference
DD_LO,    DD_HI    =    10.0,     30.0   # °F – dewpoint depression thresholds

# Gate: MIX contribution ramps 0→1 as transport wind goes 10→25 kt
TW_GATE_LO, TW_GATE_HI = 10.0, 25.0      # kt

KEEP_TOTAL_WEIGHT_CONSTANT = False

W_MIX   = 0.25
W_TWSPD = 0.45
W_SKY   = 0.15
W_DD    = 0.15

# ─────────────────────────────────────────────────────────────────────────────
#  Transport wind pressure levels
#  Ordered low-altitude → high-altitude (high mb → low mb).
#  The 10 m wind is the implicit Level 0 (surface anchor).
# ─────────────────────────────────────────────────────────────────────────────
TRANSPORT_LEVELS_MB = [950, 925, 900, 875, 850, 825, 800, 750, 700]
N_PRS_LEVELS        = len(TRANSPORT_LEVELS_MB)   # 9

# Approximate AGL height of the 10 m wind observation (metres)
SURFACE_ANCHOR_M = 10.0

# ─────────────────────────────────────────────────────────────────────────────
#  Colorado bounding box
# ─────────────────────────────────────────────────────────────────────────────
CO_LAT_MIN, CO_LAT_MAX =  37.0,  41.2
CO_LON_MIN, CO_LON_MAX = -109.5, -102.0

# ─────────────────────────────────────────────────────────────────────────────
#  File-system / cache setup
# ─────────────────────────────────────────────────────────────────────────────
HERBIE_DIR = Path(os.environ.get("HERBIE_DATA_DIR", "/tmp/herbie"))
HERBIE_DIR.mkdir(parents=True, exist_ok=True)

_CACHE: dict = {"ts": 0.0, "png": None, "meta": None}

# ─────────────────────────────────────────────────────────────────────────────
#  Unit helpers
# ─────────────────────────────────────────────────────────────────────────────
M_TO_FT  = 3.28084
MS_TO_KT = 1.94384

def _k_to_f(k: np.ndarray) -> np.ndarray:
    return (k - 273.15) * 9.0 / 5.0 + 32.0

# ─────────────────────────────────────────────────────────────────────────────
#  HRRR cycle detection
# ─────────────────────────────────────────────────────────────────────────────
def _now_utc_hour_naive() -> datetime:
    return datetime.utcnow().replace(minute=0, second=0, microsecond=0)

def _find_latest_cycle(max_lookback: int = 6) -> datetime:
    base = _now_utc_hour_naive()
    for h in range(max_lookback + 1):
        dt = base - timedelta(hours=h)
        try:
            H = Herbie(dt, model="hrrr", product="sfc", fxx=0,
                       save_dir=str(HERBIE_DIR))
            _ = H.inventory()
            return dt
        except Exception:
            continue
    logger.warning("No recent HRRR cycle found; falling back to current hour.")
    return base

# ─────────────────────────────────────────────────────────────────────────────
#  Field-fetch helpers
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_field(cycle: datetime, product: str, search: str) -> xr.Dataset:
    """Fetch one GRIB field from HRRR; normalize to xr.Dataset."""
    H = Herbie(cycle, model="hrrr", product=product, fxx=0,
               save_dir=str(HERBIE_DIR), overwrite=True)
    result = H.xarray(search, remove_grib=True)
    if isinstance(result, list):
        result = result[0] if result else xr.Dataset()
    if isinstance(result, xr.DataArray):
        result = result.to_dataset(name=result.name or "var")
    return result

def _first_var_values(ds: xr.Dataset) -> np.ndarray:
    for v in ds.data_vars:
        return np.asarray(ds[v].values, dtype=np.float32)
    raise ValueError("Empty dataset – check searchString.")

# ─────────────────────────────────────────────────────────────────────────────
#  Colorado subsetting
# ─────────────────────────────────────────────────────────────────────────────
def _co_mask(lat2d: np.ndarray, lon2d: np.ndarray) -> np.ndarray:
    lon_std = np.where(lon2d > 180.0, lon2d - 360.0, lon2d)
    return (
        (lat2d >= CO_LAT_MIN) & (lat2d <= CO_LAT_MAX) &
        (lon_std >= CO_LON_MIN) & (lon_std <= CO_LON_MAX)
    )

def _bounding_slices(mask: np.ndarray):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0 = int(np.where(rows)[0][0]);  r1 = int(np.where(rows)[0][-1])
    c0 = int(np.where(cols)[0][0]);  c1 = int(np.where(cols)[0][-1])
    return slice(r0, r1 + 1), slice(c0, c1 + 1)

# ─────────────────────────────────────────────────────────────────────────────
#  Thickness-weighted transport wind
# ─────────────────────────────────────────────────────────────────────────────
def _compute_transport_wind(
    u10m:    np.ndarray,   # (ny, nx)  m s⁻¹  – 10 m zonal wind
    v10m:    np.ndarray,   # (ny, nx)  m s⁻¹  – 10 m meridional wind
    u_prs:   np.ndarray,   # (N, ny, nx)  m s⁻¹  – U at N pressure levels, low→high AGL
    v_prs:   np.ndarray,   # (N, ny, nx)  m s⁻¹  – V at same levels
    hgt_prs: np.ndarray,   # (N, ny, nx)  m MSL  – geopotential height at same levels
    orog:    np.ndarray,   # (ny, nx)  m MSL  – terrain height
    hpbl:    np.ndarray,   # (ny, nx)  m AGL  – planetary boundary layer height
) -> tuple:
    """
    Compute the HPBL-coupled thickness-weighted mean transport wind.

    Method
    ------
    1. Build a 10-level stack (10m surface anchor + 9 pressure levels),
       sorted low → high AGL.

    2. Heights AGL for pressure levels:
         h_agl = HGT_level_msl - terrain_msl
       Levels below terrain (h_agl ≤ 0) are excluded automatically.

    3. Validity rule — "include if midpoint is below HPBL":
         midpoint[0]  = mean(0 m, h_agl[0])          surface anchor midpoint
         midpoint[i]  = mean(h_agl[i-1], h_agl[i])   pressure level midpoints
       Level i is valid if midpoint[i] < HPBL  AND  h_agl[i] > 0.

    4. Thickness weighting (midpoint rule):
         lower_bound[0]  = 0 m
         lower_bound[i]  = (h_agl[i-1] + h_agl[i]) / 2   for i ≥ 1
         upper_bound[i]  = (h_agl[i] + h_agl[i+1]) / 2   for interior
         upper_bound[-1] = HPBL                            top of layer
       All bounds are clipped to [0, HPBL].
       thickness[i] = upper_bound[i] - lower_bound[i]  → zeroed if invalid.

    5. U_mean = Σ(U[i] × thickness[i]) / Σ(thickness[i])
       V_mean = same
       speed  = √(U_mean² + V_mean²) × MS_TO_KT

    Fully vectorized — no Python loops over grid points.

    Fallback
    --------
    On the rare grid point where HPBL is so shallow that no level
    qualifies, the raw 10 m wind speed is used instead.

    Returns
    -------
    trspd_kt  : (ny, nx) transport wind speed in knots
    u_mean_ms : (ny, nx) mean U component in m s⁻¹  (diagnostic)
    v_mean_ms : (ny, nx) mean V component in m s⁻¹  (diagnostic)
    """
    N, ny, nx = u_prs.shape

    # ── Heights AGL for pressure levels ──────────────────────────────────────
    # orog broadcast: (1, ny, nx) → subtract from each level
    hgt_agl_prs = hgt_prs - orog[np.newaxis, :, :]      # (N, ny, nx), may be < 0

    # ── Full level stack: surface anchor (level 0) + pressure levels ─────────
    # h_agl shape: (N+1, ny, nx)
    h_anchor = np.full((1, ny, nx), SURFACE_ANCHOR_M, dtype=np.float32)
    h_agl    = np.concatenate([h_anchor, hgt_agl_prs], axis=0)

    # u_all, v_all shape: (N+1, ny, nx)
    u_all = np.concatenate([u10m[np.newaxis, :, :], u_prs], axis=0)
    v_all = np.concatenate([v10m[np.newaxis, :, :], v_prs], axis=0)

    # ── Midpoint AGL for each level ───────────────────────────────────────────
    # midpoint[0] = mean(0 m, h_agl[0])   — between ground and surface anchor
    # midpoint[i] = mean(h_agl[i-1], h_agl[i])  for i ≥ 1
    h_below      = np.concatenate([
        np.zeros((1, ny, nx), dtype=np.float32),
        h_agl[:-1]
    ], axis=0)                                            # (N+1, ny, nx)
    midpoint_agl = (h_below + h_agl) / 2.0               # (N+1, ny, nx)

    # ── Validity mask ─────────────────────────────────────────────────────────
    hpbl_3d = hpbl[np.newaxis, :, :]                      # broadcast (1, ny, nx)
    valid   = (h_agl > 0.0) & (midpoint_agl < hpbl_3d)   # (N+1, ny, nx)

    # ── Layer boundaries (midpoint rule, clipped to [0, HPBL]) ───────────────
    # Lower boundary: 0 for level 0; midpoint to previous level for others
    h_lower = np.concatenate([
        np.zeros((1, ny, nx), dtype=np.float32),
        (h_agl[:-1] + h_agl[1:]) / 2.0
    ], axis=0)                                            # (N+1, ny, nx)

    # Upper boundary: midpoint to next level for interior; HPBL for top
    h_upper = np.concatenate([
        (h_agl[:-1] + h_agl[1:]) / 2.0,
        hpbl_3d                                           # HPBL as ceiling
    ], axis=0)                                            # (N+1, ny, nx)

    # Clip to [0, HPBL] so integration never exceeds BL
    h_lower = np.clip(h_lower, 0.0, hpbl_3d)
    h_upper = np.clip(h_upper, 0.0, hpbl_3d)

    # Layer thickness; negative (inverted bounds at shallow BL) → 0
    dz = np.maximum(h_upper - h_lower, 0.0)              # (N+1, ny, nx)
    dz = np.where(valid, dz, 0.0)                         # zero invalid levels

    # ── Thickness-weighted mean ───────────────────────────────────────────────
    dz_total  = dz.sum(axis=0)                            # (ny, nx)
    no_layer  = dz_total < 1.0                            # fallback mask

    u_mean = (u_all * dz).sum(axis=0) / np.where(no_layer, 1.0, dz_total)
    v_mean = (v_all * dz).sum(axis=0) / np.where(no_layer, 1.0, dz_total)

    # Fallback to raw 10m wind where BL is too shallow for any level
    u_mean = np.where(no_layer, u10m, u_mean)
    v_mean = np.where(no_layer, v10m, v_mean)

    trspd_kt = (np.sqrt(u_mean**2 + v_mean**2) * MS_TO_KT).astype(np.float32)

    if no_layer.any():
        logger.debug(
            "LLTI transport wind: %d grid points had no valid BL level "
            "(HPBL too shallow); 10m wind used as fallback.",
            int(no_layer.sum()),
        )

    return trspd_kt, u_mean.astype(np.float32), v_mean.astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
#  LLTI core algorithm  (pure numpy – no GFE dependencies)
# ─────────────────────────────────────────────────────────────────────────────
def _normalize(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

def _gate_by_wind(trspd: np.ndarray) -> np.ndarray:
    frac = (trspd - TW_GATE_LO) / max(TW_GATE_HI - TW_GATE_LO, 1e-6)
    return np.clip(frac, 0.0, 1.0)

def compute_llti(
    mix_ft:   np.ndarray,
    trspd_kt: np.ndarray,
    sky_pct:  np.ndarray,
    t_f:      np.ndarray,
    td_f:     np.ndarray,
) -> np.ndarray:
    """
    Compute LLTI on a 2-D grid.  Returns float32 array in 0–100.
    Direct translation of the GFE Smart Tool algorithm.
    """
    dd = np.clip(t_f - td_f, 0.0, None)

    s_mix   = _normalize(mix_ft,   MIX_LO,   MIX_HI)
    s_twspd = _normalize(trspd_kt, TWSPD_LO, TWSPD_HI)
    s_sky   = np.clip((SKY_REF - sky_pct) / max(SKY_REF, 1e-6), 0.0, 1.0)
    s_dd    = _normalize(dd, DD_LO, DD_HI)

    g         = _gate_by_wind(trspd_kt)
    s_mix_eff = s_mix * g

    if KEEP_TOTAL_WEIGHT_CONSTANT:
        W_MIX_eff = W_MIX * g
        Wsum      = W_MIX_eff + W_TWSPD + W_SKY + W_DD
        score01   = np.clip(
            (W_MIX_eff * s_mix + W_TWSPD * s_twspd + W_SKY * s_sky + W_DD * s_dd)
            / np.maximum(Wsum, 1e-6),
            0.0, 1.0,
        )
    else:
        score01 = np.clip(
            W_MIX * s_mix_eff + W_TWSPD * s_twspd + W_SKY * s_sky + W_DD * s_dd,
            0.0, 1.0,
        )

    out = (score01 * 100.0).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=100.0, neginf=0.0)

# ─────────────────────────────────────────────────────────────────────────────
#  Main data-fetch pipeline
# ─────────────────────────────────────────────────────────────────────────────
def fetch_llti_grid() -> tuple:
    """
    Download all HRRR fields, compute the HPBL-coupled transport wind,
    and run the LLTI algorithm over Colorado.

    Returns
    -------
    lat2d   : (ny, nx) latitude  (°N)
    lon2d   : (ny, nx) longitude (°E, −180…180)
    llti2d  : (ny, nx) LLTI 0–100
    meta    : dict with cycle, stats, and diagnostic fields
    """
    cycle = _find_latest_cycle()
    logger.info("LLTI: using HRRR cycle %s UTC", cycle)

    # ── Surface fields ────────────────────────────────────────────────────────
    logger.info("LLTI: fetching HPBL …")
    ds_hpbl = _fetch_field(cycle, "sfc", ":HPBL:surface:")

    logger.info("LLTI: fetching terrain height (HGT surface) …")
    ds_orog = _fetch_field(cycle, "sfc", ":HGT:surface:")

    logger.info("LLTI: fetching 10m U wind …")
    ds_u10  = _fetch_field(cycle, "sfc", ":UGRD:10 m above ground:")

    logger.info("LLTI: fetching 10m V wind …")
    ds_v10  = _fetch_field(cycle, "sfc", ":VGRD:10 m above ground:")

    logger.info("LLTI: fetching TMP 2m …")
    ds_t2m  = _fetch_field(cycle, "sfc", ":TMP:2 m above ground:")

    logger.info("LLTI: fetching DPT 2m …")
    ds_dpt  = _fetch_field(cycle, "sfc", ":DPT:2 m above ground:")

    logger.info("LLTI: fetching TCDC …")
    ds_tcc  = _fetch_field(cycle, "sfc", ":TCDC:entire atmosphere:")

    # Shared lat/lon grid from any sfc field
    lat2d_full = np.asarray(ds_t2m["latitude"].values,  dtype=np.float32)
    lon2d_full = np.asarray(ds_t2m["longitude"].values, dtype=np.float32)

    # ── Colorado subset ───────────────────────────────────────────────────────
    mask     = _co_mask(lat2d_full, lon2d_full)
    rsl, csl = _bounding_slices(mask)

    def co(arr: np.ndarray) -> np.ndarray:
        return arr[rsl, csl]

    lat2d = co(lat2d_full)
    lon2d = co(np.where(lon2d_full > 180.0, lon2d_full - 360.0, lon2d_full))

    hpbl_m  = co(_first_var_values(ds_hpbl))
    orog_m  = co(_first_var_values(ds_orog))
    u10m    = co(_first_var_values(ds_u10))
    v10m    = co(_first_var_values(ds_v10))
    t2m_k   = co(_first_var_values(ds_t2m))
    dpt_k   = co(_first_var_values(ds_dpt))
    tcc_pct = co(_first_var_values(ds_tcc))

    mix_ft = hpbl_m * M_TO_FT
    t_f    = _k_to_f(t2m_k)
    td_f   = _k_to_f(dpt_k)

    # ── Pressure-level fields for transport wind (27 fetches) ─────────────────
    # Each level needs U, V, and geopotential height (HGT).
    # HGT is in metres MSL; subtracting OROG gives metres AGL.
    u_prs_list   = []
    v_prs_list   = []
    hgt_prs_list = []

    for mb in TRANSPORT_LEVELS_MB:
        logger.info("LLTI: fetching U/V/HGT at %d mb …", mb)
        u_prs_list.append(  co(_first_var_values(_fetch_field(cycle, "prs", f":UGRD:{mb} mb:"))))
        v_prs_list.append(  co(_first_var_values(_fetch_field(cycle, "prs", f":VGRD:{mb} mb:"))))
        hgt_prs_list.append(co(_first_var_values(_fetch_field(cycle, "prs", f":HGT:{mb} mb:"))))

    # Stack: (N_PRS_LEVELS, ny, nx) ordered 950 mb → 700 mb (low → high AGL)
    u_prs   = np.stack(u_prs_list,   axis=0).astype(np.float32)
    v_prs   = np.stack(v_prs_list,   axis=0).astype(np.float32)
    hgt_prs = np.stack(hgt_prs_list, axis=0).astype(np.float32)

    # ── HPBL-coupled thickness-weighted transport wind ────────────────────────
    logger.info("LLTI: computing thickness-weighted transport wind …")
    trspd_kt, u_mean_ms, v_mean_ms = _compute_transport_wind(
        u10m=u10m, v10m=v10m,
        u_prs=u_prs, v_prs=v_prs,
        hgt_prs=hgt_prs,
        orog=orog_m,
        hpbl=hpbl_m,
    )

    # ── LLTI ─────────────────────────────────────────────────────────────────
    llti2d = compute_llti(mix_ft, trspd_kt, tcc_pct, t_f, td_f)

    meta = {
        "cycle_utc":                cycle.strftime("%Y-%m-%dT%H:%M") + "Z",
        "fxx":                      0,
        "model":                    "HRRR",
        "llti_min":                 float(np.nanmin(llti2d)),
        "llti_max":                 float(np.nanmax(llti2d)),
        "llti_mean":                float(np.nanmean(llti2d)),
        "grid_shape":               list(llti2d.shape),
        "transport_wind_method":    "HPBL-coupled thickness-weighted mean",
        "transport_wind_levels_mb": TRANSPORT_LEVELS_MB,
        "transport_wind_anchor":    "10m wind (surface)",
        "hpbl_mean_ft":             float(np.nanmean(hpbl_m) * M_TO_FT),
        "trspd_mean_kt":            float(np.nanmean(trspd_kt)),
    }

    logger.info(
        "LLTI: min=%.0f  mean=%.0f  max=%.0f  |  "
        "HPBL mean=%.0f ft  |  TransWind mean=%.1f kt",
        meta["llti_min"], meta["llti_mean"], meta["llti_max"],
        meta["hpbl_mean_ft"], meta["trspd_mean_kt"],
    )

    return lat2d, lon2d, llti2d, meta

# ─────────────────────────────────────────────────────────────────────────────
#  Colormap
# ─────────────────────────────────────────────────────────────────────────────
LLTI_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "llti",
    [
        (0.00, "#006400"),
        (0.25, "#7FFF00"),
        (0.50, "#FFD700"),
        (0.70, "#FF8C00"),
        (0.85, "#FF0000"),
        (1.00, "#8B0000"),
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
#  PNG renderer
# ─────────────────────────────────────────────────────────────────────────────
def render_llti_png(
    lat2d:  np.ndarray,
    lon2d:  np.ndarray,
    llti2d: np.ndarray,
    meta:   dict,
    dpi:    int = 130,
) -> bytes:
    fig, ax = plt.subplots(figsize=(9, 6), dpi=dpi)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    pcm = ax.pcolormesh(
        lon2d, lat2d, llti2d,
        cmap=LLTI_CMAP, vmin=0, vmax=100,
        shading="auto",
    )

    cbar = fig.colorbar(pcm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("LLTI (0–100)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    for thresh in (25, 50, 75):
        cbar.ax.axhline(thresh, color="white", linewidth=0.6, linestyle="--", alpha=0.5)

    ax.set_title(
        f"HRRR Low-Level Turbulence Index (LLTI)\n"
        f"Cycle: {meta['cycle_utc']}  |  F{meta['fxx']:02d}  |  Colorado",
        color="white", fontsize=10, pad=8,
    )
    ax.set_xlabel("Longitude (°E)", color="white", fontsize=8)
    ax.set_ylabel("Latitude (°N)",  color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    ax.text(
        0.01, 0.02,
        f"min={meta['llti_min']:.0f}   mean={meta['llti_mean']:.0f}   max={meta['llti_max']:.0f}",
        transform=ax.transAxes, color="white", fontsize=7,
        bbox=dict(facecolor="#1a1a2e", alpha=0.8, edgecolor="none", pad=3),
    )

    ax.text(
        0.99, 0.98,
        f"Transport wind: HPBL-coupled thickness-weighted\n"
        f"Anchor: 10m  +  {', '.join(str(m) for m in TRANSPORT_LEVELS_MB)} mb\n"
        f"HPBL mean: {meta['hpbl_mean_ft']:.0f} ft  |  "
        f"TrWnd mean: {meta['trspd_mean_kt']:.1f} kt",
        transform=ax.transAxes,
        color="#aaaaaa", fontsize=6,
        ha="right", va="top",
        bbox=dict(facecolor="#1a1a2e", alpha=0.7, edgecolor="none", pad=2),
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────────────────────────────────────
#  Cached public entry point  (PNG + home-page metadata)
# ─────────────────────────────────────────────────────────────────────────────
def get_llti_cached(ttl_seconds: int = 600) -> tuple:
    """
    Return (png_bytes, meta_dict), cached for ttl_seconds.
    HRRR updates hourly; 10-minute default TTL is appropriate.
    """
    now = time.time()
    if _CACHE["png"] is None or (now - _CACHE["ts"]) > ttl_seconds:
        logger.info("LLTI cache miss – fetching fresh data …")
        lat, lon, llti, meta = fetch_llti_grid()
        _CACHE["png"]  = render_llti_png(lat, lon, llti, meta)
        _CACHE["meta"] = meta
        _CACHE["ts"]   = now
    return _CACHE["png"], _CACHE["meta"]


# ─────────────────────────────────────────────────────────────────────────────
#  Point-data output  (used by the interactive Leaflet map)
# ─────────────────────────────────────────────────────────────────────────────

# Separate cache keyed by (cycle_utc, fxx) so each hour is cached independently
_POINTS_CACHE: dict = {}

# Grid stride: HRRR ~3 km → every 2nd point ≈ 6 km, ~25k points over Colorado
_STRIDE = 2


def _cat_from_llti(score: float) -> int:
    """Map LLTI 0-100 score to 4-level risk category."""
    if score >= 75: return 3   # high
    if score >= 50: return 2   # moderate
    if score >= 25: return 1   # low
    return 0                   # negligible


def fetch_llti_points(cycle_utc: str, fxx: int = 1) -> dict:
    """
    Fetch HRRR fields for the requested cycle + forecast hour and return
    LLTI as a JSON-serialisable point list, matching the format used by
    /api/winds/colorado, /api/froude/colorado, etc.

    Parameters
    ----------
    cycle_utc : str   e.g. "2026-03-01T18:00Z"
    fxx       : int   forecast hour 0-18

    Returns
    -------
    dict with keys: points, valid_utc, cycle_utc, fxx, point_count,
                    cell_size_deg, model, transport_wind_method
    """
    # Parse cycle string → naive datetime for Herbie
    cycle_dt = datetime.strptime(cycle_utc.replace("Z", ""), "%Y-%m-%dT%H:%M")
    valid_dt  = cycle_dt + timedelta(hours=fxx)
    valid_utc = valid_dt.strftime("%Y-%m-%dT%H:%M") + "Z"

    logger.info("LLTI points: cycle=%s  fxx=%d", cycle_utc, fxx)

    # ── fetch helper ──────────────────────────────────────────────────────────
    # Key design decision: use ONE shared save_dir per (cycle, fxx, product).
    # This means all sfc fields share the same directory, and if two fields
    # happen to occupy the same byte range (same hash), they legitimately
    # reuse the same subset file — which is correct behavior.
    # Using per-field subdirectories causes the opposite problem: Herbie looks
    # for a file in the field-specific dir, doesn't find it, tries to download,
    # and cfgrib then can't open it because the session state is inconsistent.
    _dirs: dict = {}
    def fetch(product: str, search: str) -> xr.Dataset:
        if product not in _dirs:
            d = HERBIE_DIR / f"llti_{cycle_dt.strftime('%Y%m%d%H')}_{fxx:02d}_{product}"
            d.mkdir(parents=True, exist_ok=True)
            _dirs[product] = d
        H = Herbie(cycle_dt, model="hrrr", product=product, fxx=fxx,
                   save_dir=str(_dirs[product]), overwrite=False)
        result = H.xarray(search, remove_grib=False)
        if isinstance(result, list):
            result = result[0] if result else xr.Dataset()
        if isinstance(result, xr.DataArray):
            result = result.to_dataset(name=result.name or "var")
        return result

    # ── Surface fields — each searchstring called exactly once ───────────────
    # NOTE: terrain height in HRRR sfc is ":HGT:surface:" not ":OROG:surface:"
    # OROG is not a standard HRRR sfc field; HGT:surface is geopotential height
    # at the surface = MSL terrain height in metres, which is what we need.
    ds_t2m  = fetch("sfc", ":TMP:2 m above ground:")
    ds_hpbl = fetch("sfc", ":HPBL:surface:")
    ds_orog = fetch("sfc", ":HGT:surface:")          # terrain height MSL (m)
    ds_u10  = fetch("sfc", ":UGRD:10 m above ground:")
    ds_v10  = fetch("sfc", ":VGRD:10 m above ground:")
    ds_dpt  = fetch("sfc", ":DPT:2 m above ground:")
    ds_tcc  = fetch("sfc", ":TCDC:entire atmosphere:")

    # lat/lon extracted from the already-fetched TMP dataset — no second fetch
    lat2d_full = np.asarray(ds_t2m["latitude"].values,  dtype=np.float32)
    lon2d_full = np.asarray(ds_t2m["longitude"].values, dtype=np.float32)

    mask     = _co_mask(lat2d_full, lon2d_full)
    rsl, csl = _bounding_slices(mask)

    def co(arr): return arr[rsl, csl]

    lat2d = co(lat2d_full)
    lon2d = co(np.where(lon2d_full > 180.0, lon2d_full - 360.0, lon2d_full))

    hpbl_m  = co(_first_var_values(ds_hpbl))
    orog_m  = co(_first_var_values(ds_orog))
    u10m    = co(_first_var_values(ds_u10))
    v10m    = co(_first_var_values(ds_v10))
    t2m_k   = co(_first_var_values(ds_t2m))
    dpt_k   = co(_first_var_values(ds_dpt))
    tcc_pct = co(_first_var_values(ds_tcc))

    mix_ft = hpbl_m * M_TO_FT
    t_f    = _k_to_f(t2m_k)
    td_f   = _k_to_f(dpt_k)

    # Pressure-level fields for transport wind
    u_prs_list, v_prs_list, hgt_prs_list = [], [], []
    for mb in TRANSPORT_LEVELS_MB:
        u_prs_list.append(  co(_first_var_values(fetch("prs", f":UGRD:{mb} mb:"))))
        v_prs_list.append(  co(_first_var_values(fetch("prs", f":VGRD:{mb} mb:"))))
        hgt_prs_list.append(co(_first_var_values(fetch("prs", f":HGT:{mb} mb:"))))

    u_prs   = np.stack(u_prs_list,   axis=0).astype(np.float32)
    v_prs   = np.stack(v_prs_list,   axis=0).astype(np.float32)
    hgt_prs = np.stack(hgt_prs_list, axis=0).astype(np.float32)

    trspd_kt, _, _ = _compute_transport_wind(
        u10m=u10m, v10m=v10m,
        u_prs=u_prs, v_prs=v_prs,
        hgt_prs=hgt_prs, orog=orog_m, hpbl=hpbl_m,
    )

    llti2d = compute_llti(mix_ft, trspd_kt, tcc_pct, t_f, td_f)

    # ── Build point list (subsampled for map performance) ─────────────────────
    ny, nx = llti2d.shape
    points = []
    for iy in range(0, ny, _STRIDE):
        for ix in range(0, nx, _STRIDE):
            score = float(llti2d[iy, ix])
            points.append({
                "lat":      round(float(lat2d[iy, ix]),  3),
                "lon":      round(float(lon2d[iy, ix]),  3),
                "llti":     round(score, 1),
                "cat":      _cat_from_llti(score),
                "mix_ft":   round(float(mix_ft[iy, ix]),   0),
                "trspd_kt": round(float(trspd_kt[iy, ix]), 1),
                "sky_pct":  round(float(tcc_pct[iy, ix]),  0),
                "dd_f":     round(float(np.clip(
                                t_f[iy, ix] - td_f[iy, ix], 0, None)), 1),
            })

    return {
        "points":               points,
        "valid_utc":            valid_utc,
        "cycle_utc":            cycle_utc,
        "fxx":                  fxx,
        "point_count":          len(points),
        "cell_size_deg":        0.05,
        "model":                "HRRR",
        "transport_wind_method": "HPBL-coupled thickness-weighted mean",
        "transport_wind_levels_mb": TRANSPORT_LEVELS_MB,
    }


def get_llti_points_cached(cycle_utc: str, fxx: int = 1,
                            ttl_seconds: int = 600) -> dict:
    """
    Return point data for the map, cached per (cycle_utc, fxx).
    """
    key = (cycle_utc, fxx)
    now = time.time()
    entry = _POINTS_CACHE.get(key)
    if entry is None or (now - entry["ts"]) > ttl_seconds:
        logger.info("LLTI points cache miss – cycle=%s fxx=%d", cycle_utc, fxx)
        data = fetch_llti_points(cycle_utc, fxx)
        _POINTS_CACHE[key] = {"ts": now, "data": data}
    return _POINTS_CACHE[key]["data"]
