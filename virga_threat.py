"""
virga_threat.py
===============
CONUS Virga Potential for model-viewer.

Adapted from virga.py to support:
  - Full CONUS domain (not clipped to Colorado)
  - Both RAP13 and HRRR via herbie_model / prs_product params
  - Parameterized pressure levels (HRRR has 550/650/750 mb; RAP13 does not)
  - RAP13 uses RH directly; HRRR computes RH from T+Td
  - Missing levels degrade gracefully (warning, not crash)
  - fetch_virga_arrays() returning numpy 2D arrays for the renderer

Science
-------
1. Upper saturated layer (700-500 mb): mean RH >= 80% over 200 mb depth
2. Max 100 mb RH decrease in column (850-500 mb): evaporation zone
3. virga_pct = RH decrease masked by upper cloud presence (0-100%)
"""

import gc
import os
import time
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pygrib
from herbie import Herbie

logger = logging.getLogger(__name__)

HERBIE_DIR = Path(os.environ.get("HERBIE_DATA_DIR", "/tmp/herbie"))
HERBIE_DIR.mkdir(parents=True, exist_ok=True)

# Loose CONUS clip
LAT_MIN, LAT_MAX = 20.0, 55.0
LON_MIN, LON_MAX = -130.0, -60.0

# Subsampling defaults
DEFAULT_STEP_HRRR = 6
DEFAULT_STEP_RAP  = 2

# HRRR has intermediate levels; RAP13 standard pressure levels only
LEVELS_HRRR = [500, 550, 600, 650, 700, 750, 800, 850]
LEVELS_RAP  = [500, 600, 700, 800, 850]

# GRIB searches — RAP13 uses RH instead of DPT
SEARCH_HRRR = r"(?:TMP|DPT|UGRD|VGRD):(?:500|550|600|650|700|750|800|850) mb"
SEARCH_RAP  = r"(?:TMP|RH|UGRD|VGRD):(?:500|600|700|800|850) mb"

_CACHE: dict    = {}
_CLIP_IDX: dict = {}

NAME_MAP = {
    "Temperature":           "T",
    "Dew point temperature": "Td",
    "Relative humidity":     "RH",   # RAP13 provides RH instead of Td
    "U component of wind":   "U",
    "V component of wind":   "V",
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _get_clip_idx(lat2d, lon2d):
    key = lat2d.shape
    if key in _CLIP_IDX:
        return _CLIP_IDX[key]
    mask = ((lat2d >= LAT_MIN) & (lat2d <= LAT_MAX) &
            (lon2d >= LON_MIN) & (lon2d <= LON_MAX))
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("CONUS clip bounds do not intersect model grid.")
    idx = (int(rows[0]), int(rows[-1]) + 1,
           int(cols[0]), int(cols[-1]) + 1)
    _CLIP_IDX[key] = idx
    return idx


def _clip(arr, clip_idx, step):
    r0, r1, c0, c1 = clip_idx
    return arr[r0:r1, c0:c1][::step, ::step].astype(np.float32)


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def _rh_from_td(T_K, Td_K):
    """August-Roche-Magnus approximation."""
    T_C  = T_K  - 273.15
    Td_C = Td_K - 273.15
    e_T  = 6.112 * np.exp(17.67 * T_C  / (T_C  + 243.5))
    e_Td = 6.112 * np.exp(17.67 * Td_C / (Td_C + 243.5))
    return np.clip(100.0 * e_Td / e_T, 0.0, 100.0)


# ---------------------------------------------------------------------------
# GRIB read — single pass, lat/lon computed once
# ---------------------------------------------------------------------------

def _read_fields(path, levels_expected, step):
    """
    Single-pass pygrib read.
    - First match pays the pyproj cost for lat/lon; all others use grb.values.
    - Accepts either Td (HRRR) or RH directly (RAP13).
    - Returns rh dict keyed by level, ready for the science functions.
    """
    levels_set = frozenset(levels_expected)
    T  = {}
    Td = {}
    RH = {}
    U  = {}
    V  = {}
    lat_out = lon_out = clip_idx = None

    grbs = pygrib.open(str(path))
    try:
        for grb in grbs:
            if grb.typeOfLevel != "isobaricInhPa":
                continue
            lev = grb.level
            if lev not in levels_set:
                continue
            var = NAME_MAP.get(grb.name)
            if var is None:
                continue

            if clip_idx is None:
                data, lat2d, lon2d = grb.data()
                lon2d = np.where(lon2d > 180, lon2d - 360, lon2d)
                clip_idx = _get_clip_idx(lat2d, lon2d)
                r0, r1, c0, c1 = clip_idx
                lat_out = lat2d[r0:r1, c0:c1][::step, ::step].astype(np.float32)
                lon_out = lon2d[r0:r1, c0:c1][::step, ::step].astype(np.float32)
                del lat2d, lon2d
            else:
                data = grb.values

            small = _clip(data, clip_idx, step)
            del data

            if   var == "T":  T[lev]  = small
            elif var == "Td": Td[lev] = small
            elif var == "RH": RH[lev] = small
            elif var == "U":  U[lev]  = small
            elif var == "V":  V[lev]  = small
    finally:
        grbs.close()
        gc.collect()

    # Build unified RH dict — prefer direct RH, fall back to T+Td conversion
    rh_out = {}
    for lev in levels_expected:
        if lev in RH:
            rh_out[lev] = RH[lev]
        elif lev in T and lev in Td:
            rh_out[lev] = _rh_from_td(T[lev], Td[lev])

    usable = sorted(rh_out.keys())

    missing = [l for l in levels_expected if l not in rh_out]
    if missing:
        logger.warning("virga: RH unavailable at levels %s — skipping.", missing)
    if not usable:
        raise ValueError(
            "virga: no usable RH levels found. "
            "Check search string and NAME_MAP for this model/product."
        )

    return lat_out, lon_out, rh_out, U, V, usable


# ---------------------------------------------------------------------------
# Core compute
# ---------------------------------------------------------------------------

def _compute(herbie_model, prs_product, cycle, fxx, step):
    levels = LEVELS_HRRR if herbie_model == "hrrr" else LEVELS_RAP
    search = SEARCH_HRRR if herbie_model == "hrrr" else SEARCH_RAP

    H = Herbie(cycle, model=herbie_model, product=prs_product,
               fxx=fxx, save_dir=str(HERBIE_DIR), overwrite=False)
    result = H.download(searchString=search)
    path = Path(result) if result else None
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"virga: download failed {herbie_model}/{prs_product} "
            f"{cycle} F{fxx:02d}"
        )

    lat, lon, rh, U, V, usable = _read_fields(path, levels, step)
    shape = lat.shape

    # 1. Upper saturated layer (700-500 mb): mean RH >= 80% over 200 mb depth
    upper_levels = [l for l in usable if l <= 700]
    max_upper_rh = np.zeros(shape, dtype=np.float32)
    for lev_top in upper_levels:
        window = [l for l in upper_levels if lev_top <= l <= lev_top + 200]
        if len(window) < 2:
            continue
        mean_rh = np.mean([rh[l] for l in window], axis=0)
        max_upper_rh = np.maximum(max_upper_rh, mean_rh)
    upper_cloud = max_upper_rh >= 80.0

    # 2. Max 100 mb RH decrease in column (850-500 mb)
    max_rh_decrease = np.zeros(shape, dtype=np.float32)
    for lev_bot in sorted(usable, reverse=True):
        lev_top = lev_bot - 100
        if lev_top not in rh:
            continue
        decrease = (rh[lev_bot] - rh[lev_top]).astype(np.float32)
        max_rh_decrease = np.maximum(max_rh_decrease, decrease)

    # 3. Mask by upper cloud presence
    virga_pct = np.where(upper_cloud,
                         np.clip(max_rh_decrease, 0.0, 100.0),
                         0.0).astype(np.float32)
    return lat, lon, virga_pct


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_virga_arrays(herbie_model: str,
                       prs_product: str,
                       cycle_dt: datetime,
                       fxx: int,
                       subsample_step=None):
    """
    Return (lat2d, lon2d, virga_pct2d) for the renderer.
    virga_pct: 0=none, ~20=low, ~40=moderate, ~60=high, ~80+=extreme.
    """
    if subsample_step is None:
        subsample_step = DEFAULT_STEP_HRRR if herbie_model == "hrrr" else DEFAULT_STEP_RAP

    cycle = cycle_dt.replace(tzinfo=None)
    cache_key = (herbie_model, prs_product, cycle.isoformat(), fxx, subsample_step)
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and (now - cached["ts"]) < 600:
        return cached["lat"], cached["lon"], cached["virga"]

    lat, lon, virga = _compute(herbie_model, prs_product, cycle, fxx, subsample_step)
    _CACHE[cache_key] = {"ts": now, "lat": lat, "lon": lon, "virga": virga}
    return lat, lon, virga
