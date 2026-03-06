"""
virga_threat.py
===============
CONUS Virga Potential + Virga Turbulence Potential for model-viewer.

Supports:
  - Full CONUS domain
  - RAP13 and HRRR
  - Graceful degradation when some levels are missing
  - Returns two 2D arrays:
      1) virga_potential        (0-100)
      2) virga_turb_potential   (0-100)

Scientific idea
---------------
A) Virga Potential
   Measures whether the column supports precipitation forming aloft and
   evaporating into a drier layer below.

   Ingredients:
     1. Moist source layer aloft (roughly 700-500 mb)
     2. Adequate source-layer depth
     3. Drying beneath the source layer

B) Virga Turbulence Potential
   Measures whether plausible virga also overlaps a favorable environment
   for downdraft / turbulence impacts.

   Ingredients:
     1. Virga potential
     2. Steep sub-cloud lapse rate proxy (temperature drop with pressure)
     3. Wind speed in/near sub-cloud layer (momentum / mixing proxy)

Notes
-----
- RAP13 provides RH directly on pressure levels.
- HRRR computes RH from T and Td.
- This is still a proxy product, not a full microphysics model.
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

# Pressure levels available
LEVELS_HRRR = [500, 550, 600, 650, 700, 750, 800, 850]
LEVELS_RAP  = [500, 600, 700, 800, 850]

# Search strings
SEARCH_HRRR = r"(?:TMP|DPT|UGRD|VGRD):(?:500|550|600|650|700|750|800|850) mb"
SEARCH_RAP  = r"(?:TMP|RH|UGRD|VGRD):(?:500|600|700|800|850) mb"

_CACHE: dict = {}
_CLIP_IDX: dict = {}

NAME_MAP = {
    "Temperature":           "T",
    "Dew point temperature": "Td",
    "Relative humidity":     "RH",
    "U component of wind":   "U",
    "V component of wind":   "V",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def _score_linear(x, lo, hi):
    """
    Linear score from 0 to 1.
    x <= lo -> 0
    x >= hi -> 1
    """
    if hi <= lo:
        raise ValueError(f"Invalid score range: lo={lo}, hi={hi}")
    return _clip01((x - lo) / (hi - lo))


def _get_clip_idx(lat2d, lon2d):
    key = lat2d.shape
    if key in _CLIP_IDX:
        return _CLIP_IDX[key]

    mask = (
        (lat2d >= LAT_MIN) & (lat2d <= LAT_MAX) &
        (lon2d >= LON_MIN) & (lon2d <= LON_MAX)
    )
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


def _weighted_layer_mean(field_dict, levels):
    """
    Pressure-thickness weighted mean over a layer defined by pressure levels.
    Uses trapezoidal averaging between adjacent pressure levels.

    Parameters
    ----------
    field_dict : dict[level -> 2D array]
    levels     : iterable of pressure levels in mb

    Returns
    -------
    mean_field : 2D array or None
    total_dp   : total layer depth in mb actually represented
    """
    levels = sorted([l for l in levels if l in field_dict])
    if len(levels) < 2:
        return None, 0

    numer = None
    total_dp = 0.0

    for i in range(len(levels) - 1):
        p1 = levels[i]
        p2 = levels[i + 1]
        dp = float(p2 - p1)
        if dp <= 0:
            continue

        seg_mean = 0.5 * (field_dict[p1] + field_dict[p2])
        if numer is None:
            numer = seg_mean * dp
        else:
            numer += seg_mean * dp
        total_dp += dp

    if numer is None or total_dp <= 0:
        return None, 0

    return (numer / total_dp).astype(np.float32), total_dp


def _layer_max(field_dict, levels):
    levels = [l for l in levels if l in field_dict]
    if not levels:
        return None
    return np.maximum.reduce([field_dict[l] for l in levels]).astype(np.float32)


def _layer_min(field_dict, levels):
    levels = [l for l in levels if l in field_dict]
    if not levels:
        return None
    return np.minimum.reduce([field_dict[l] for l in levels]).astype(np.float32)


def _wind_speed(U, V, lev):
    if lev not in U or lev not in V:
        return None
    return np.sqrt(U[lev] * U[lev] + V[lev] * V[lev]).astype(np.float32)


def _rh_from_td(T_K, Td_K):
    """August-Roche-Magnus approximation."""
    T_C  = T_K  - 273.15
    Td_C = Td_K - 273.15
    e_T  = 6.112 * np.exp(17.67 * T_C  / (T_C  + 243.5))
    e_Td = 6.112 * np.exp(17.67 * Td_C / (Td_C + 243.5))
    return np.clip(100.0 * e_Td / e_T, 0.0, 100.0).astype(np.float32)


# ---------------------------------------------------------------------------
# GRIB read
# ---------------------------------------------------------------------------

def _read_fields(path, levels_expected, step):
    """
    Single-pass pygrib read.
    Returns:
      lat_out, lon_out, rh_dict, T_dict, U_dict, V_dict, usable_levels
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

    # Build unified RH dict
    rh_out = {}
    for lev in levels_expected:
        if lev in RH:
            rh_out[lev] = RH[lev]
        elif lev in T and lev in Td:
            rh_out[lev] = _rh_from_td(T[lev], Td[lev])

    usable = sorted(rh_out.keys())

    missing_rh = [l for l in levels_expected if l not in rh_out]
    if missing_rh:
        logger.warning("virga: RH unavailable at levels %s", missing_rh)

    if not usable:
        raise ValueError(
            "virga: no usable RH levels found. "
            "Check search string and NAME_MAP for this model/product."
        )

    return lat_out, lon_out, rh_out, T, U, V, usable


# ---------------------------------------------------------------------------
# Science
# ---------------------------------------------------------------------------

def _compute_virga_potential(rh, usable, shape):
    """
    Compute base virga potential from:
      - source-layer moisture (roughly 700-500 mb)
      - source-layer depth
      - drying below source layer

    Returns
    -------
    virga_potential : 0-100
    diagnostics     : dict of intermediate fields
    """

    # --- Source layer: prioritize 700-500 mb, but use what exists there.
    source_levels = [l for l in usable if 500 <= l <= 700]

    source_mean_rh, source_depth = _weighted_layer_mean(rh, source_levels)
    source_max_rh = _layer_max(rh, source_levels)

    if source_mean_rh is None or source_max_rh is None:
        logger.warning("virga: insufficient source-layer RH for virga potential.")
        zeros = np.zeros(shape, dtype=np.float32)
        return zeros, {
            "source_mean_rh": zeros,
            "source_max_rh": zeros,
            "source_depth": 0.0,
            "subcloud_mean_rh": zeros,
            "subcloud_min_rh": zeros,
            "dryness_delta": zeros,
        }

    # --- Dry layer below source region: use 850-700 when possible
    subcloud_levels = [l for l in usable if 700 <= l <= 850]
    subcloud_mean_rh, subcloud_depth = _weighted_layer_mean(rh, subcloud_levels)
    subcloud_min_rh = _layer_min(rh, subcloud_levels)

    if subcloud_mean_rh is None or subcloud_min_rh is None:
        logger.warning("virga: insufficient sub-cloud RH for virga potential.")
        zeros = np.zeros(shape, dtype=np.float32)
        return zeros, {
            "source_mean_rh": source_mean_rh,
            "source_max_rh": source_max_rh,
            "source_depth": source_depth,
            "subcloud_mean_rh": zeros,
            "subcloud_min_rh": zeros,
            "dryness_delta": zeros,
        }

    # --- Scores
    # Moist source layer:
    # Broad support if weighted mean RH is high enough
    source_mean_score = _score_linear(source_mean_rh, 70.0, 90.0)

    # Peak RH allowance for thinner/moist pockets that still make precip
    source_peak_score = _score_linear(source_max_rh, 82.0, 95.0)

    # Depth score: prefer at least ~150-200 mb of represented source layer
    depth_score_scalar = float(_score_linear(np.array(source_depth), 100.0, 200.0))
    depth_score = np.full(shape, depth_score_scalar, dtype=np.float32)

    # Dry air below
    dryness_delta = (source_mean_rh - subcloud_mean_rh).astype(np.float32)
    dryness_delta_score = _score_linear(dryness_delta, 8.0, 30.0)

    subcloud_dry_score = _score_linear(70.0 - subcloud_mean_rh, 0.0, 30.0)
    subcloud_min_score = _score_linear(60.0 - subcloud_min_rh, 0.0, 30.0)

    # Blend
    source_score = (
        0.55 * source_mean_score +
        0.25 * source_peak_score +
        0.20 * depth_score
    )

    dry_score = (
        0.50 * dryness_delta_score +
        0.30 * subcloud_dry_score +
        0.20 * subcloud_min_score
    )

    virga_score01 = _clip01(source_score * dry_score)
    virga_potential = (100.0 * virga_score01).astype(np.float32)

    diagnostics = {
        "source_mean_rh": source_mean_rh.astype(np.float32),
        "source_max_rh": source_max_rh.astype(np.float32),
        "source_depth": source_depth,
        "subcloud_mean_rh": subcloud_mean_rh.astype(np.float32),
        "subcloud_min_rh": subcloud_min_rh.astype(np.float32),
        "dryness_delta": dryness_delta.astype(np.float32),
    }
    return virga_potential, diagnostics


def _compute_turbulence_modifier(T, U, V, usable, shape):
    """
    Compute turbulence-relevant modifier from:
      - sub-cloud lapse-rate proxy using temperature drop between 700 and 850 mb
      - sub-cloud/source-layer wind speed proxy using 700/750/800 mb winds

    Returns
    -------
    turb_modifier_01 : 0-1
    diagnostics      : dict
    """

    # --- Lapse-rate proxy from 700 to 850 mb
    # Large temp increase downward (T850 - T700) implies steeper lapse rate.
    if 700 in T and 850 in T:
        dT_700_850 = (T[850] - T[700]).astype(np.float32)  # K over 150 mb
        lapse_score = _score_linear(dT_700_850, 6.0, 16.0)
    elif 700 in T and 800 in T:
        dT_700_800 = (T[800] - T[700]).astype(np.float32)  # K over 100 mb
        lapse_score = _score_linear(dT_700_800, 4.0, 11.0)
    else:
        logger.warning("virga: insufficient temperature levels for lapse proxy.")
        lapse_score = np.zeros(shape, dtype=np.float32)

    # --- Wind proxy from sub-cloud / source-base levels
    candidate_winds = []
    for lev in (700, 750, 800):
        wspd = _wind_speed(U, V, lev)
        if wspd is not None:
            candidate_winds.append(wspd)

    if candidate_winds:
        mean_wspd = np.mean(candidate_winds, axis=0).astype(np.float32)
        wind_score = _score_linear(mean_wspd, 15.0, 40.0)
    else:
        logger.warning("virga: insufficient wind levels for turbulence proxy.")
        mean_wspd = np.zeros(shape, dtype=np.float32)
        wind_score = np.zeros(shape, dtype=np.float32)

    # Blend turbulence ingredients
    turb_modifier_01 = _clip01(
        0.60 * lapse_score +
        0.40 * wind_score
    ).astype(np.float32)

    diagnostics = {
        "lapse_score": lapse_score.astype(np.float32),
        "mean_wspd": mean_wspd.astype(np.float32),
        "wind_score": wind_score.astype(np.float32),
    }
    return turb_modifier_01, diagnostics


# ---------------------------------------------------------------------------
# Core compute
# ---------------------------------------------------------------------------

def _compute(herbie_model, prs_product, cycle, fxx, step):
    levels = LEVELS_HRRR if herbie_model == "hrrr" else LEVELS_RAP
    search = SEARCH_HRRR if herbie_model == "hrrr" else SEARCH_RAP

    H = Herbie(
        cycle,
        model=herbie_model,
        product=prs_product,
        fxx=fxx,
        save_dir=str(HERBIE_DIR),
        overwrite=False,
    )

    result = H.download(searchString=search)
    path = Path(result) if result else None
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"virga: download failed {herbie_model}/{prs_product} "
            f"{cycle} F{fxx:02d}"
        )

    lat, lon, rh, T, U, V, usable = _read_fields(path, levels, step)
    shape = lat.shape

    virga_potential, virga_diag = _compute_virga_potential(rh, usable, shape)
    turb_modifier_01, turb_diag = _compute_turbulence_modifier(T, U, V, usable, shape)

    # Turbulence potential is virga potential * turbulence modifier
    virga_turb_potential = (
        virga_potential * turb_modifier_01
    ).astype(np.float32)

    return lat, lon, virga_potential, virga_turb_potential


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_virga_arrays(
    herbie_model: str,
    prs_product: str,
    cycle_dt: datetime,
    fxx: int,
    subsample_step=None,
):
    """
    Return:
      lat2d, lon2d, virga_potential2d, virga_turb_potential2d

    Suggested interpretation:
      0-20   : little / none
      20-40  : low
      40-60  : moderate
      60-80  : high
      80-100 : very high

    Notes:
      - virga_potential diagnoses evaporative / virga-favorable structure
      - virga_turb_potential further weights that by lapse rate + wind
    """
    if subsample_step is None:
        subsample_step = (
            DEFAULT_STEP_HRRR if herbie_model == "hrrr" else DEFAULT_STEP_RAP
        )

    cycle = cycle_dt.replace(tzinfo=None)
    cache_key = (herbie_model, prs_product, cycle.isoformat(), fxx, subsample_step)
    now = time.time()

    cached = _CACHE.get(cache_key)
    if cached and (now - cached["ts"]) < 600:
        return (
            cached["lat"],
            cached["lon"],
            cached["virga_potential"],
            cached["virga_turb_potential"],
        )

    lat, lon, virga_potential, virga_turb_potential = _compute(
        herbie_model, prs_product, cycle, fxx, subsample_step
    )

    _CACHE[cache_key] = {
        "ts": now,
        "lat": lat,
        "lon": lon,
        "virga_potential": virga_potential,
        "virga_turb_potential": virga_turb_potential,
    }

    return lat, lon, virga_potential, virga_turb_potential
