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
   Broad environmental support for virga:
     - moist source layer aloft
     - adequate source-layer depth
     - dry air beneath

B) Virga Turbulence Potential
   Legacy-style targeted signal for aviation impacts:
     - upper moist-layer gate
     - maximum 100 mb RH decrease in the column
     - wind near the highest rapid-drying layer
     - lapse-rate proxy

Notes
-----
- RAP13 provides RH directly on pressure levels.
- HRRR computes RH from T and Td.
- This is a proxy product, not a full cloud microphysics model.
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
LEVELS_HRRR = [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
LEVELS_RAP  = [300, 400, 500, 600, 700, 800, 850, 900, 950, 1000]

# Search strings
SEARCH_HRRR = r"(?:TMP|DPT|UGRD|VGRD):(?:300|350|400|450|500|550|600|650|700|750|800|850|900|950|1000) mb"
SEARCH_RAP  = r"(?:TMP|RH|UGRD|VGRD):(?:300|400|500|600|700|800|850|900|950|1000) mb"

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


def _mean_wind_levels(U, V, levels, shape):
    winds = []
    for lev in levels:
        wspd = _wind_speed(U, V, lev)
        if wspd is not None:
            winds.append(wspd)
    if not winds:
        return np.zeros(shape, dtype=np.float32)
    return np.mean(winds, axis=0).astype(np.float32)


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
# Broad virga potential
# ---------------------------------------------------------------------------

def _compute_virga_potential(rh, usable, shape):
    """
    Compute broad environmental virga potential from:
      - source-layer moisture
      - source-layer depth
      - dry air beneath
    """
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

    subcloud_levels = [l for l in usable if 700 <= l <= 850]
    subcloud_mean_rh, _ = _weighted_layer_mean(rh, subcloud_levels)
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

    source_mean_score = _score_linear(source_mean_rh, 70.0, 90.0)
    source_peak_score = _score_linear(source_max_rh, 82.0, 95.0)
    depth_score_scalar = float(_score_linear(np.array(source_depth), 100.0, 200.0))
    depth_score = np.full(shape, depth_score_scalar, dtype=np.float32)

    dryness_delta = (source_mean_rh - subcloud_mean_rh).astype(np.float32)
    dryness_delta_score = _score_linear(dryness_delta, 8.0, 30.0)
    subcloud_dry_score = _score_linear(70.0 - subcloud_mean_rh, 0.0, 30.0)
    subcloud_min_score = _score_linear(60.0 - subcloud_min_rh, 0.0, 30.0)

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


# ---------------------------------------------------------------------------
# Legacy-style targeted turbulence logic
# ---------------------------------------------------------------------------

def _compute_legacy_turbulence_terms(rh, U, V, usable, shape):
    """
    Reincorporates legacy logic:

      1. Find maximum 200 mb mean RH aloft using sliding windows
         (similar spirit to old 700-500 / 600-400 / 500-300 search)

      2. Find maximum 100 mb RH decrease through the column

      3. Identify the highest layer where RH decrease exceeds threshold,
         and assign wind near the top of that drying layer
         (legacy cloud-base-ish wind proxy)

    Returns diagnostics used in final virga_turb_potential.
    """
    zeros = np.zeros(shape, dtype=np.float32)

    # --- Upper moist-layer gate: use sliding 200 mb windows aloft
    upper_window_starts = [300, 400, 500]
    max_rh_upper = np.zeros(shape, dtype=np.float32)

    found_upper_window = False
    for p_top in upper_window_starts:
        window = [l for l in usable if p_top <= l <= p_top + 200]
        if len(window) < 2:
            continue
        mean_rh, total_dp = _weighted_layer_mean(rh, window)
        if mean_rh is None or total_dp < 150:
            continue
        max_rh_upper = np.maximum(max_rh_upper, mean_rh)
        found_upper_window = True

    if not found_upper_window:
        logger.warning("virga: insufficient upper-level windows for moist-layer gate.")

    # --- Default wind field if no sharp drying layer is found:
    # mean 700-500 wind, in the spirit of the old script
    default_cloud_base_wind = _mean_wind_levels(U, V, [500, 600, 700], shape)

    # --- Max 100 mb RH decrease and wind at highest qualifying drying layer
    max_rh_delta_100mb = np.zeros(shape, dtype=np.float32)
    cloud_base_wind = default_cloud_base_wind.copy()

    # Sort bottom-up so later overwrites correspond to higher levels,
    # matching the old "highest rapid-drying layer" idea.
    sorted_levels = sorted(usable, reverse=True)

    # Need lev_top and lev_bot = lev_top + 100
    candidate_tops = sorted([l for l in usable if (l + 100) in rh], reverse=True)

    for lev_top in candidate_tops:
        lev_bot = lev_top + 100

        rh_delta_here = (rh[lev_top] - rh[lev_bot]).astype(np.float32)
        max_rh_delta_100mb = np.maximum(max_rh_delta_100mb, rh_delta_here)

        # Wind near top of drying layer:
        # old script used a ~50 mb mean centered near the top;
        # with sparse pressure levels, approximate with mean of lev_top and lev_top+50 if available.
        wind_levels = [lev_top]
        if (lev_top + 50) in usable:
            wind_levels.append(lev_top + 50)

        mean_wind_here = _mean_wind_levels(U, V, wind_levels, shape)

        # Overwrite where this layer qualifies as rapid drying
        cloud_base_wind = np.where(
            rh_delta_here >= 40.0,
            mean_wind_here,
            cloud_base_wind
        ).astype(np.float32)

    # Mask RH delta by upper moist layer gate, like legacy logic
    max_rh_delta_masked = np.where(max_rh_upper >= 80.0, max_rh_delta_100mb, 0.0).astype(np.float32)

    diagnostics = {
        "max_rh_upper": max_rh_upper.astype(np.float32),
        "max_rh_delta_100mb": max_rh_delta_100mb.astype(np.float32),
        "max_rh_delta_masked": max_rh_delta_masked.astype(np.float32),
        "cloud_base_wind": cloud_base_wind.astype(np.float32),
        "default_cloud_base_wind": default_cloud_base_wind.astype(np.float32),
    }
    return diagnostics


def _compute_lapse_proxy(T, shape):
    """
    Simple lapse-rate proxy from pressure-level temperatures.
    Large temperature increase downward implies steeper lapse rate.
    """
    if 700 in T and 850 in T:
        dT = (T[850] - T[700]).astype(np.float32)   # K over 150 mb
        lapse_score = _score_linear(dT, 6.0, 16.0)
        return lapse_score.astype(np.float32), dT
    elif 700 in T and 800 in T:
        dT = (T[800] - T[700]).astype(np.float32)   # K over 100 mb
        lapse_score = _score_linear(dT, 4.0, 11.0)
        return lapse_score.astype(np.float32), dT
    else:
        logger.warning("virga: insufficient temperature levels for lapse proxy.")
        zeros = np.zeros(shape, dtype=np.float32)
        return zeros, zeros


def _compute_virga_turbulence_potential(virga_potential, rh, T, U, V, usable, shape):
    """
    Final turbulence-oriented product:
      - legacy max RH delta logic
      - upper moist-layer gate
      - cloud-base-ish wind
      - lapse proxy
      - weak influence from broad virga potential
    """
    legacy = _compute_legacy_turbulence_terms(rh, U, V, usable, shape)
    lapse_score, lapse_proxy = _compute_lapse_proxy(T, shape)

    max_rh_delta_masked = legacy["max_rh_delta_masked"]
    cloud_base_wind = legacy["cloud_base_wind"]

    rh_delta_score = _score_linear(max_rh_delta_masked, 20.0, 50.0)
    wind_score = _score_linear(cloud_base_wind, 20.0, 50.0)
    virga_env_score = _score_linear(virga_potential, 15.0, 60.0)

    # Main idea:
    # - RH delta is the backbone
    # - wind and lapse modulate the aviation relevance
    # - broad virga potential lightly supports continuity / sanity
    turb_score01 = _clip01(
        0.45 * rh_delta_score +
        0.25 * wind_score +
        0.20 * lapse_score +
        0.10 * virga_env_score
    )

    # Hard gate: if no upper moist layer, kill the signal
    upper_gate = legacy["max_rh_upper"] >= 75.0
    turb_score01 = np.where(upper_gate, turb_score01, 0.0).astype(np.float32)

    virga_turb_potential = (100.0 * turb_score01).astype(np.float32)

    diagnostics = {
        "max_rh_upper": legacy["max_rh_upper"],
        "max_rh_delta_100mb": legacy["max_rh_delta_100mb"],
        "max_rh_delta_masked": max_rh_delta_masked,
        "cloud_base_wind": cloud_base_wind,
        "lapse_proxy": lapse_proxy.astype(np.float32),
        "lapse_score": lapse_score.astype(np.float32),
        "rh_delta_score": rh_delta_score.astype(np.float32),
        "wind_score": wind_score.astype(np.float32),
        "virga_env_score": virga_env_score.astype(np.float32),
    }
    return virga_turb_potential, diagnostics


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

    virga_potential, _ = _compute_virga_potential(rh, usable, shape)
    virga_turb_potential, _ = _compute_virga_turbulence_potential(
        virga_potential, rh, T, U, V, usable, shape
    )

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
    """
    if subsample_step is None:
        subsample_step = (
            DEFAULT_STEP_HRRR if herbie_model == "hrrr" else DEFAULT_STEP_RAP
        )

    cycle = cycle_dt.replace(tzinfo=None)
    cache_key = (
        herbie_model,
        prs_product,
        cycle.isoformat(),
        fxx,
        subsample_step,
    )
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
