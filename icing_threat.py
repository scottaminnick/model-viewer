"""
icing_threat.py
===============
Winter Icing Threat Index for model-viewer.

Adapted from icing_threat_index_hrrr.py to support:
  - Full CONUS domain (not clipped to Colorado)
  - Both RAP13 and HRRR via herbie_model / herbie_product params
  - fetch_icing_arrays() returning numpy 2D arrays for the renderer
  - fetch_icing() returning a point list for debugging / API use

Ingredients (core score)
  Saturation   min(RH850, RH700)
  Ascent       min(VVEL850, VVEL700)
  Convergence  -div(U850,V850)
  Upslope      850 mb wind sector + speed modifier

DGZ additions
  DGZ Saturation  proxy saturation in DGZ window (~-18 to -12 C)
  DGZ Lift        proxy ascent in DGZ layer
  DGZ Band Index  dgz_sat * dgz_lift
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

# Loose CONUS clip (slightly wider than renderer bounds to avoid edge artifacts)
LAT_MIN, LAT_MAX = 20.0, 55.0
LON_MIN, LON_MAX = -130.0, -60.0

# Subsampling: HRRR 3km full CONUS is ~1800x1060; step=6 -> ~300x177 manageable
# RAP13 full CONUS is ~337x451; step=2 -> ~169x226
# Caller passes step via fetch_icing_arrays(subsample_step=...)
DEFAULT_STEP_HRRR = 6
DEFAULT_STEP_RAP  = 2

# GRIB search — works for both RAP awp130pgrb and HRRR prs
PRS_SEARCH = r"(?:RH|UGRD|VGRD|VVEL|TMP):(?:850|700|750) mb"

# Scoring weights
W_SAT    = 0.45
W_ASCENT = 0.35
W_CONV   = 0.20

# Upslope modifiers
UPSLOPE_FRONT_RANGE = 0.15   # FROM 045–135°
UPSLOPE_WEST_SLOPE  = 0.10   # FROM 225–315°
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

# Grid spacing after subsampling (meters) — approximate
DX_M = 6000.0
DY_M = 6000.0

# DGZ window
DGZ_TMIN_C   = -18.0
DGZ_TMAX_C   = -12.0
DGZ_RH_ON    = 85.0

# DGZ influence
DGZ_BAND_BOOST    = 0.10
DGZ_SLD_SUPPRESS  = 0.00
WARM_NOSE_BOOST   = 0.00

# In-memory cache keyed by (herbie_model, herbie_product, cycle_utc_str, fxx)
_CACHE: dict = {}
_CLIP_IDX: dict = {}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_celsius(t_k: np.ndarray) -> np.ndarray:
    return t_k - 273.15


def _get_clip_idx(lat2d: np.ndarray, lon2d: np.ndarray):
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


def _clip_and_subsample(data: np.ndarray, clip_idx, step: int) -> np.ndarray:
    r0, r1, c0, c1 = clip_idx
    return data[r0:r1, c0:c1][::step, ::step].astype(np.float32)


# ---------------------------------------------------------------------------
# GRIB download + read
# ---------------------------------------------------------------------------

def _download_subset(herbie_model: str, herbie_product: str,
                     cycle: datetime, fxx: int) -> Path:
    H = Herbie(cycle, model=herbie_model, product=herbie_product,
               fxx=fxx, save_dir=str(HERBIE_DIR), overwrite=False)
    result = H.download(searchString=PRS_SEARCH)
    p = Path(result) if result else None
    if p is None or not p.exists():
        raise FileNotFoundError(
            f"Download failed: {herbie_model}/{herbie_product} "
            f"{cycle} F{fxx:02d}"
        )
    return p


def _read_prs_fields(subset_path: Path, step: int):
    """
    Single-pass pygrib read. Returns clipped + subsampled arrays.
    """
    want = {
        ("Relative humidity",   850): "RH850",
        ("Relative humidity",   700): "RH700",
        ("U component of wind", 850): "U850",
        ("V component of wind", 850): "V850",
        ("U component of wind", 700): "U700",
        ("V component of wind", 700): "V700",
        ("Vertical velocity",   850): "VVEL850",
        ("Vertical velocity",   700): "VVEL700",
        ("Temperature",         850): "T850",
        ("Temperature",         750): "T750",
        ("Temperature",         700): "T700",
    }

    fields: dict = {}
    lat_out = lon_out = None
    clip_idx = None

    grbs = pygrib.open(str(subset_path))
    try:
        for grb in grbs:
            if grb.typeOfLevel != "isobaricInhPa":
                continue
            key = (grb.name, grb.level)
            if key not in want:
                continue

            data, lat2d, lon2d = grb.data()
            lon2d = np.where(lon2d > 180, lon2d - 360, lon2d)

            if clip_idx is None:
                clip_idx = _get_clip_idx(lat2d, lon2d)
                r0, r1, c0, c1 = clip_idx
                lat_out = lat2d[r0:r1, c0:c1][::step, ::step].astype(np.float32)
                lon_out = lon2d[r0:r1, c0:c1][::step, ::step].astype(np.float32)

            fields[want[key]] = _clip_and_subsample(data, clip_idx, step)
            del data, lat2d, lon2d
    finally:
        grbs.close()
        gc.collect()

    missing = [k for k in want.values() if k not in fields]
    if missing:
        raise ValueError(
            f"Missing prs fields: {missing}. "
            f"Check PRS_SEARCH and GRIB 'name' strings."
        )

    return (
        lat_out, lon_out,
        fields["RH850"],  fields["RH700"],
        fields["U850"],   fields["V850"],
        fields["U700"],   fields["V700"],
        fields["VVEL850"], fields["VVEL700"],
        fields["T850"],   fields["T750"],  fields["T700"],
    )


# ---------------------------------------------------------------------------
# Science functions (unchanged from your algorithm)
# ---------------------------------------------------------------------------

def _saturation_score(rh850, rh700):
    rh_min = np.minimum(rh850, rh700)
    return np.clip((rh_min - 80.0) / 20.0, 0.0, 1.0).astype(np.float32)


def _ascent_score(vvel850, vvel700):
    vv = np.minimum(vvel850, vvel700)
    strength = -vv
    weak   = -ASCENT_WEAK
    strong = -ASCENT_STRONG
    return np.clip((strength - weak) / (strong - weak), 0.0, 1.0).astype(np.float32)


def _convergence_score(u850, v850):
    du_dy, du_dx = np.gradient(u850, DY_M, DX_M)
    dv_dy, dv_dx = np.gradient(v850, DY_M, DX_M)
    convergence = -(du_dx + dv_dy)
    return np.clip((convergence - CONV_WEAK) / (CONV_STRONG - CONV_WEAK),
                   0.0, 1.0).astype(np.float32)


def _upslope_modifier(u850, v850):
    spd_kt = np.sqrt(u850**2 + v850**2) * 1.94384
    wdir = (np.degrees(np.arctan2(u850, v850)) + 360.0) % 360.0
    front_range = (wdir >= 45.0)  & (wdir <= 135.0) & (spd_kt >= UPSLOPE_SPD_KT)
    west_slope  = (wdir >= 225.0) & (wdir <= 315.0) & (spd_kt >= UPSLOPE_SPD_KT)
    mod = np.zeros_like(spd_kt, dtype=np.float32)
    mod[front_range] += UPSLOPE_FRONT_RANGE
    mod[west_slope]  += UPSLOPE_WEST_SLOPE
    return mod


def _dgz_saturation_score(t700_k, t750_k, t850_k, rh700, rh850):
    t700 = _to_celsius(t700_k)
    t750 = _to_celsius(t750_k)
    t850 = _to_celsius(t850_k)
    m700 = (t700 >= DGZ_TMIN_C) & (t700 <= DGZ_TMAX_C)
    m750 = (t750 >= DGZ_TMIN_C) & (t750 <= DGZ_TMAX_C)
    m850 = (t850 >= DGZ_TMIN_C) & (t850 <= DGZ_TMAX_C)
    rh750 = 0.5 * (rh700 + rh850)

    def rh_score(rh):
        return np.clip((rh - DGZ_RH_ON) / (100.0 - DGZ_RH_ON), 0.0, 1.0)

    s700 = np.where(m700, rh_score(rh700), 0.0)
    s750 = np.where(m750, rh_score(rh750), 0.0)
    s850 = np.where(m850, rh_score(rh850), 0.0)
    return np.maximum.reduce([s700, s750, s850]).astype(np.float32)


def _dgz_lift_score(t700_k, t750_k, t850_k, vvel700, vvel850):
    t700 = _to_celsius(t700_k)
    t750 = _to_celsius(t750_k)
    t850 = _to_celsius(t850_k)
    m700 = (t700 >= DGZ_TMIN_C) & (t700 <= DGZ_TMAX_C)
    m750 = (t750 >= DGZ_TMIN_C) & (t750 <= DGZ_TMAX_C)
    m850 = (t850 >= DGZ_TMIN_C) & (t850 <= DGZ_TMAX_C)
    vvel750 = 0.5 * (vvel700 + vvel850)

    def vv_score(vv):
        strength = -vv
        weak   = -ASCENT_WEAK
        strong = -ASCENT_STRONG
        return np.clip((strength - weak) / (strong - weak), 0.0, 1.0)

    l700 = np.where(m700, vv_score(vvel700), 0.0)
    l750 = np.where(m750, vv_score(vvel750), 0.0)
    l850 = np.where(m850, vv_score(vvel850), 0.0)
    return np.maximum.reduce([l700, l750, l850]).astype(np.float32)


def _warm_nose_flag(t700_k, t850_k):
    return (_to_celsius(t700_k) > 0.0) & (_to_celsius(t850_k) < 0.0)


def _categorize(score):
    cat = np.zeros_like(score, dtype=np.int8)
    cat[score >= CAT_YELLOW] = 1
    cat[score >= CAT_ORANGE] = 2
    cat[score >= CAT_RED]    = 3
    return cat


# ---------------------------------------------------------------------------
# Core compute
# ---------------------------------------------------------------------------

def _compute(herbie_model: str, herbie_product: str,
             cycle: datetime, fxx: int, step: int):
    """
    Download, read, and score. Returns (lat2d, lon2d, score2d, cat2d).
    """
    prs_path = _download_subset(herbie_model, herbie_product, cycle, fxx)

    (lat, lon,
     rh850, rh700,
     u850,  v850,
     u700,  v700,
     vvel850, vvel700,
     t850, t750, t700) = _read_prs_fields(prs_path, step)

    sat  = _saturation_score(rh850, rh700)
    asc  = _ascent_score(vvel850, vvel700)
    conv = _convergence_score(u850, v850)
    ups  = _upslope_modifier(u850, v850)

    dgz_sat  = _dgz_saturation_score(t700, t750, t850, rh700, rh850)
    dgz_lift = _dgz_lift_score(t700, t750, t850, vvel700, vvel850)
    dgz_band = (dgz_sat * dgz_lift).astype(np.float32)

    warm_nose = _warm_nose_flag(t700, t850)

    score = (W_SAT * sat + W_ASCENT * asc + W_CONV * conv + ups).astype(np.float32)
    score += DGZ_BAND_BOOST * dgz_band

    if DGZ_SLD_SUPPRESS > 0:
        score -= DGZ_SLD_SUPPRESS * dgz_band
    if WARM_NOSE_BOOST > 0:
        score += WARM_NOSE_BOOST * warm_nose.astype(np.float32)

    score = np.clip(score, 0.0, 1.2).astype(np.float32)
    cat   = _categorize(score)

    return lat, lon, score, cat


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_icing_arrays(herbie_model: str, herbie_product: str,
                       cycle_dt: datetime, fxx: int,
                       subsample_step: int | None = None
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (lat2d, lon2d, score2d) for the renderer.
    score2d values: 0.0 = none, ~0.35 = light, ~0.55 = moderate, ~0.75+ = heavy
    """
    if subsample_step is None:
        subsample_step = DEFAULT_STEP_HRRR if herbie_model == "hrrr" else DEFAULT_STEP_RAP

    cycle = cycle_dt.replace(tzinfo=None)
    cache_key = (herbie_model, herbie_product,
                 cycle.isoformat(), fxx, subsample_step)
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and (now - cached["ts"]) < 600:
        return cached["lat"], cached["lon"], cached["score"]

    lat, lon, score, _ = _compute(herbie_model, herbie_product,
                                  cycle, fxx, subsample_step)
    _CACHE[cache_key] = {"ts": now, "lat": lat, "lon": lon, "score": score}
    return lat, lon, score
