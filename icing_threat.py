"""
icing_threat.py  (v2)
=====================
Winter Icing Threat Index for model-viewer.

Adapted from icing_threat_index_hrrr_v2.py to support:
  - Full CONUS domain (not clipped to Colorado)
  - Both RAP13 and HRRR via herbie_model / prs_product / sfc_product params
  - fetch_icing_arrays() returning numpy 2D arrays for the renderer
  - sfc_product=None gracefully skips APCP (RAP13 fallback)

V2 upgrades over V1
  - SLW proxy:        CLWMR + RWMR hydrometeor mixing ratios
  - Temp weighting:   ascent + SLW weighted toward icing-favorable window (-18 to -5 C)
  - Precip-rate proxy: APCP 1-hr accumulation as band intensity modifier
  - DGZ band boost retained from v1

Ingredients (core score)
  Saturation   min(RH850, RH700)          weight 0.40
  Ascent       min(VVEL850, VVEL700)      weight 0.30  (temp-weighted)
  Convergence  -div(U850,V850)            weight 0.15
  SLW proxy    max(CLWMR+RWMR) layers    weight 0.15  (temp-weighted)
  Upslope      850mb sector heuristic     additive modifier
  DGZ band     dgz_sat * dgz_lift         additive boost
  APCP bonus   1-hr precip rate           additive modifier (capped)
"""

import os
import gc
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pygrib
from herbie import Herbie

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HERBIE_DIR = Path(os.environ.get("HERBIE_DATA_DIR", "/tmp/herbie"))
HERBIE_DIR.mkdir(parents=True, exist_ok=True)

# Loose CONUS clip - slightly wider than renderer bounds to avoid edge artifacts
LAT_MIN, LAT_MAX = 20.0, 55.0
LON_MIN, LON_MAX = -130.0, -60.0

# Subsampling defaults
DEFAULT_STEP_HRRR = 6   # HRRR 3km full CONUS ~1800x1060 -> ~300x177
DEFAULT_STEP_RAP  = 2   # RAP13 full CONUS ~337x451 -> ~169x226

# GRIB searches
PRS_SEARCH = r"(?:RH|UGRD|VGRD|VVEL|TMP|CLWMR|RWMR):(?:850|700|750) mb"
SFC_SEARCH = r"(?:APCP):surface"

# Scoring weights
W_SAT    = 0.40
W_ASCENT = 0.30
W_CONV   = 0.15
W_SLW    = 0.15

# Upslope modifiers
UPSLOPE_FRONT_RANGE = 0.12   # FROM 045-135 deg
UPSLOPE_WEST_SLOPE  = 0.08   # FROM 225-315 deg
UPSLOPE_SPD_KT      = 10.0

# Category thresholds
CAT_YELLOW = 0.35
CAT_ORANGE = 0.55
CAT_RED    = 0.75

# Ascent thresholds (Pa/s, negative = ascent)
ASCENT_WEAK   = -0.10
ASCENT_STRONG = -0.50

# Convergence thresholds (s^-1)
CONV_WEAK   = 0.5e-5
CONV_STRONG = 2.0e-5

# Grid spacing after subsampling (meters)
DX_M = 6000.0
DY_M = 6000.0

# DGZ window
DGZ_TMIN_C       = -18.0
DGZ_TMAX_C       = -12.0
DGZ_RH_ON        = 85.0
DGZ_BAND_BOOST   = 0.08
DGZ_SLD_SUPPRESS = 0.00

# Temperature favorability window for icing
TEMP_FAV_MIN_C    = -18.0
TEMP_FAV_MAX_C    =  -5.0
TEMP_TAPER_COLD_C = -25.0
TEMP_TAPER_WARM_C =  +1.0

# SLW proxy thresholds (kg/kg)
SLW_WEAK   = 1.0e-5
SLW_STRONG = 8.0e-5

# APCP proxy thresholds (mm) and max bonus
APCP_WEAK      = 0.5
APCP_STRONG    = 3.0
APCP_MAX_BONUS = 0.10

# In-memory caches
_CACHE: dict    = {}
_CLIP_IDX: dict = {}

# GRIB name synonyms
NAME_MAP = {
    "RH":    ["Relative humidity"],
    "U":     ["U component of wind"],
    "V":     ["V component of wind"],
    "VVEL":  ["Vertical velocity"],
    "TMP":   ["Temperature"],
    "CLWMR": ["Cloud mixing ratio", "Cloud water mixing ratio", "Cloud water"],
    "RWMR":  ["Rain mixing ratio",  "Rain water mixing ratio",  "Rain water"],
    "APCP":  ["Total Precipitation", "Total precipitation"],
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_celsius(t_k):
    return t_k - 273.15


def _get_clip_idx(lat2d, lon2d):
    shape_key = lat2d.shape
    if shape_key in _CLIP_IDX:
        return _CLIP_IDX[shape_key]
    mask = ((lat2d >= LAT_MIN) & (lat2d <= LAT_MAX) &
            (lon2d >= LON_MIN) & (lon2d <= LON_MAX))
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("CONUS clip bounds do not intersect the model grid.")
    idx = (int(rows[0]), int(rows[-1]) + 1,
           int(cols[0]), int(cols[-1]) + 1)
    _CLIP_IDX[shape_key] = idx
    return idx


def _clip_and_subsample(data, clip_idx, step):
    r0, r1, c0, c1 = clip_idx
    return data[r0:r1, c0:c1][::step, ::step].astype(np.float32)


# ---------------------------------------------------------------------------
# GRIB download + read
# ---------------------------------------------------------------------------

def _download_subset(herbie_model, herbie_product, cycle, fxx, search):
    H = Herbie(cycle, model=herbie_model, product=herbie_product,
               fxx=fxx, save_dir=str(HERBIE_DIR), overwrite=False)
    result = H.download(searchString=search)
    p = Path(result) if result else None
    if p is None or not p.exists():
        raise FileNotFoundError(
            f"Download failed: {herbie_model}/{herbie_product} "
            f"{cycle} F{fxx:02d} search={search}"
        )
    return p


def _read_prs_fields(prs_path, step):
    """
    Single-pass pygrib read. CLWMR/RWMR optional, filled with zeros if absent.
    """
    required = {
        ("RH",   850): "RH850",  ("RH",   700): "RH700",
        ("U",    850): "U850",   ("V",    850): "V850",
        ("U",    700): "U700",   ("V",    700): "V700",
        ("VVEL", 850): "VVEL850",("VVEL", 700): "VVEL700",
        ("TMP",  850): "T850",   ("TMP",  750): "T750",  ("TMP", 700): "T700",
    }
    optional = {
        ("CLWMR", 850): "CLWMR850", ("CLWMR", 750): "CLWMR750", ("CLWMR", 700): "CLWMR700",
        ("RWMR",  850): "RWMR850",  ("RWMR",  750): "RWMR750",  ("RWMR",  700): "RWMR700",
    }

    out = {}
    lat_out = lon_out = clip_idx = None

    def _concept(grb):
        if grb.typeOfLevel != "isobaricInhPa": return None
        level = int(grb.level)
        if level not in (700, 750, 850): return None
        for concept, names in NAME_MAP.items():
            if grb.name in names: return concept, level
        return None

    grbs = pygrib.open(str(prs_path))
    try:
        for grb in grbs:
            key = _concept(grb)
            if key is None: continue
            data, lat2d, lon2d = grb.data()
            lon2d = np.where(lon2d > 180, lon2d - 360, lon2d)
            if clip_idx is None:
                clip_idx = _get_clip_idx(lat2d, lon2d)
                r0, r1, c0, c1 = clip_idx
                lat_out = lat2d[r0:r1, c0:c1][::step, ::step].astype(np.float32)
                lon_out = lon2d[r0:r1, c0:c1][::step, ::step].astype(np.float32)
            if key in required:
                out[required[key]] = _clip_and_subsample(data, clip_idx, step)
            elif key in optional:
                out[optional[key]] = _clip_and_subsample(data, clip_idx, step)
            del data, lat2d, lon2d
    finally:
        grbs.close()
        gc.collect()

    missing = [v for v in required.values() if v not in out]
    if missing:
        raise ValueError(f"Missing required prs fields: {missing}. Check PRS_SEARCH and NAME_MAP.")

    for v in optional.values():
        if v not in out:
            out[v] = np.zeros_like(out["RH850"], dtype=np.float32)

    return (
        lat_out, lon_out,
        out["RH850"],    out["RH700"],
        out["U850"],     out["V850"],
        out["U700"],     out["V700"],
        out["VVEL850"],  out["VVEL700"],
        out["T850"],     out["T750"],     out["T700"],
        out["CLWMR850"], out["CLWMR750"], out["CLWMR700"],
        out["RWMR850"],  out["RWMR750"],  out["RWMR700"],
    )


def _read_sfc_apcp(sfc_path, step, ref_shape):
    grbs = pygrib.open(str(sfc_path))
    try:
        for grb in grbs:
            if grb.typeOfLevel != "surface": continue
            if grb.name not in NAME_MAP["APCP"]: continue
            data, lat2d, lon2d = grb.data()
            lon2d = np.where(lon2d > 180, lon2d - 360, lon2d)
            clip_idx = _get_clip_idx(lat2d, lon2d)
            result = _clip_and_subsample(data, clip_idx, step)
            del data, lat2d, lon2d
            return result
    finally:
        grbs.close()
        gc.collect()
    return np.zeros(ref_shape, dtype=np.float32)


# ---------------------------------------------------------------------------
# Science functions
# ---------------------------------------------------------------------------

def _saturation_score(rh850, rh700):
    return np.clip((np.minimum(rh850, rh700) - 80.0) / 20.0, 0.0, 1.0).astype(np.float32)


def _ascent_score(vvel850, vvel700):
    strength = -np.minimum(vvel850, vvel700)
    return np.clip((strength - (-ASCENT_WEAK)) / ((-ASCENT_STRONG) - (-ASCENT_WEAK)),
                   0.0, 1.0).astype(np.float32)


def _convergence_score(u850, v850):
    du_dy, du_dx = np.gradient(u850, DY_M, DX_M)
    dv_dy, dv_dx = np.gradient(v850, DY_M, DX_M)
    convergence = -(du_dx + dv_dy)
    return np.clip((convergence - CONV_WEAK) / (CONV_STRONG - CONV_WEAK),
                   0.0, 1.0).astype(np.float32)


def _upslope_modifier(u850, v850):
    spd_kt = np.sqrt(u850**2 + v850**2) * 1.94384
    wdir = (np.degrees(np.arctan2(u850, v850)) + 360.0) % 360.0
    mod = np.zeros_like(spd_kt, dtype=np.float32)
    mod[(wdir >= 45.0)  & (wdir <= 135.0) & (spd_kt >= UPSLOPE_SPD_KT)] += UPSLOPE_FRONT_RANGE
    mod[(wdir >= 225.0) & (wdir <= 315.0) & (spd_kt >= UPSLOPE_SPD_KT)] += UPSLOPE_WEST_SLOPE
    return mod


def _temp_favorability_weight(t850_k, t750_k, t700_k):
    def w(tc):
        cold = np.clip((tc - TEMP_TAPER_COLD_C) / (TEMP_FAV_MIN_C - TEMP_TAPER_COLD_C), 0.0, 1.0)
        warm = np.clip((TEMP_TAPER_WARM_C - tc) / (TEMP_TAPER_WARM_C - TEMP_FAV_MAX_C), 0.0, 1.0)
        core = (tc >= TEMP_FAV_MIN_C) & (tc <= TEMP_FAV_MAX_C)
        return np.where(core, 1.0, np.minimum(cold, warm)).astype(np.float32)
    return np.maximum.reduce([w(_to_celsius(t850_k)),
                               w(_to_celsius(t750_k)),
                               w(_to_celsius(t700_k))]).astype(np.float32)


def _dgz_band_index(t850_k, t750_k, t700_k, rh850, rh700, vvel850, vvel700):
    t850, t750, t700 = _to_celsius(t850_k), _to_celsius(t750_k), _to_celsius(t700_k)
    rh750   = 0.5 * (rh700 + rh850)
    vvel750 = 0.5 * (vvel700 + vvel850)

    def rh_s(rh):
        return np.clip((rh - DGZ_RH_ON) / (100.0 - DGZ_RH_ON), 0.0, 1.0)

    def vv_s(vv):
        return np.clip((-vv - (-ASCENT_WEAK)) / ((-ASCENT_STRONG) - (-ASCENT_WEAK)), 0.0, 1.0)

    m850 = (t850 >= DGZ_TMIN_C) & (t850 <= DGZ_TMAX_C)
    m750 = (t750 >= DGZ_TMIN_C) & (t750 <= DGZ_TMAX_C)
    m700 = (t700 >= DGZ_TMIN_C) & (t700 <= DGZ_TMAX_C)

    dgz_sat  = np.maximum.reduce([np.where(m850, rh_s(rh850), 0.0),
                                   np.where(m750, rh_s(rh750), 0.0),
                                   np.where(m700, rh_s(rh700), 0.0)]).astype(np.float32)
    dgz_lift = np.maximum.reduce([np.where(m850, vv_s(vvel850),  0.0),
                                   np.where(m750, vv_s(vvel750),  0.0),
                                   np.where(m700, vv_s(vvel700),  0.0)]).astype(np.float32)
    return (dgz_sat * dgz_lift).astype(np.float32)


def _slw_proxy_score(clw850, clw750, clw700, rw850, rw750, rw700, temp_weight):
    slw = np.maximum.reduce([clw850+rw850, clw750+rw750, clw700+rw700]).astype(np.float32)
    slw_score = np.clip((slw - SLW_WEAK) / (SLW_STRONG - SLW_WEAK), 0.0, 1.0)
    return (slw_score * temp_weight).astype(np.float32)


def _apcp_bonus(apcp_1hr):
    rate_score = np.clip((apcp_1hr - APCP_WEAK) / (APCP_STRONG - APCP_WEAK), 0.0, 1.0)
    return (APCP_MAX_BONUS * rate_score).astype(np.float32)


def _categorize(score):
    cat = np.zeros_like(score, dtype=np.int8)
    cat[score >= CAT_YELLOW] = 1
    cat[score >= CAT_ORANGE] = 2
    cat[score >= CAT_RED]    = 3
    return cat


# ---------------------------------------------------------------------------
# Core compute
# ---------------------------------------------------------------------------

def _compute(herbie_model, prs_product, sfc_product, cycle, fxx, step):
    prs_path = _download_subset(herbie_model, prs_product, cycle, fxx, PRS_SEARCH)

    (lat, lon,
     rh850, rh700,
     u850,  v850,
     u700,  v700,
     vvel850, vvel700,
     t850, t750, t700,
     clw850, clw750, clw700,
     rw850,  rw750,  rw700) = _read_prs_fields(prs_path, step)

    if sfc_product is not None:
        try:
            sfc_path = _download_subset(herbie_model, sfc_product, cycle, fxx, SFC_SEARCH)
            apcp_1hr = _read_sfc_apcp(sfc_path, step, lat.shape)
        except Exception:
            apcp_1hr = np.zeros_like(lat, dtype=np.float32)
    else:
        apcp_1hr = np.zeros_like(lat, dtype=np.float32)

    sat      = _saturation_score(rh850, rh700)
    asc      = _ascent_score(vvel850, vvel700)
    conv     = _convergence_score(u850, v850)
    ups      = _upslope_modifier(u850, v850)
    temp_wt  = _temp_favorability_weight(t850, t750, t700)
    dgz_band = _dgz_band_index(t850, t750, t700, rh850, rh700, vvel850, vvel700)
    slw      = _slw_proxy_score(clw850, clw750, clw700, rw850, rw750, rw700, temp_wt)
    apcp_bon = _apcp_bonus(apcp_1hr)

    score = (
        W_SAT    * sat          +
        W_ASCENT * (asc * temp_wt) +
        W_CONV   * conv         +
        W_SLW    * slw          +
        ups
    ).astype(np.float32)

    score += DGZ_BAND_BOOST * dgz_band + apcp_bon
    if DGZ_SLD_SUPPRESS > 0:
        score -= DGZ_SLD_SUPPRESS * dgz_band

    return lat, lon, np.clip(score, 0.0, 1.3).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_icing_arrays(herbie_model: str,
                       prs_product: str,
                       cycle_dt: datetime,
                       fxx: int,
                       sfc_product=None,
                       subsample_step=None):
    """
    Return (lat2d, lon2d, score2d) for the renderer.
    score: 0.0=none, ~0.35=light, ~0.55=moderate, ~0.75+=heavy.

    sfc_product: surface product string for APCP download.
                 Pass None to skip (score slightly lower but still valid).
    """
    if subsample_step is None:
        subsample_step = DEFAULT_STEP_HRRR if herbie_model == "hrrr" else DEFAULT_STEP_RAP

    cycle = cycle_dt.replace(tzinfo=None)
    cache_key = (herbie_model, prs_product, str(sfc_product),
                 cycle.isoformat(), fxx, subsample_step)
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and (now - cached["ts"]) < 600:
        return cached["lat"], cached["lon"], cached["score"]

    lat, lon, score = _compute(herbie_model, prs_product, sfc_product,
                                cycle, fxx, subsample_step)
    _CACHE[cache_key] = {"ts": now, "lat": lat, "lon": lon, "score": score}
    return lat, lon, score
