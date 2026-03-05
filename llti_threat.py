"""
llti_threat.py
==============
CONUS Low-Level Turbulence Index for model-viewer.

Adapted from llti.py to support:
  - Full CONUS domain (not clipped to Colorado)
  - Both RAP13 and HRRR via herbie_model / sfc_product / prs_product params
  - Parameterized transport wind levels (RAP13 skips 875/825 mb)
  - Stripped PNG renderer (model-viewer handles rendering)
  - fetch_llti_arrays() returning numpy 2D arrays for the renderer

Algorithm (preserved exactly from GFE Smart Tool)
  score = W_MIX   * s_mix_eff   (gated by transport wind speed)
        + W_TWSPD * s_twspd
        + W_SKY   * s_sky        (inverted: clear sky -> higher score)
        + W_DD    * s_dd         (dry air -> higher score)
  LLTI  = clip(score, 0, 1) * 100

Variable mapping
  MixHgt    -> HPBL (m -> ft)                            sfc product
  TransWind -> HPBL-coupled thickness-weighted mean wind sfc + prs products
  Sky       -> TCDC entire atmosphere (%)                sfc product
  T         -> TMP 2m above ground (K -> F)              sfc product
  Td        -> DPT 2m above ground (K -> F)              sfc product
"""

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
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

# Transport wind pressure levels, low -> high AGL
# RAP13 conservatively skips 875/825 mb (uncertain availability)
TRANSPORT_LEVELS_HRRR = [950, 925, 900, 875, 850, 825, 800, 750, 700]
TRANSPORT_LEVELS_RAP  = [950, 925, 900, 875, 850, 825, 800, 750, 700]

# Algorithm constants (mirror GFE tool exactly)
MIX_LO,   MIX_HI   = 5_000.0, 12_000.0   # ft
TWSPD_LO, TWSPD_HI =    20.0,     60.0   # kt
SKY_REF              =    70.0             # %
DD_LO,    DD_HI    =    10.0,     30.0   # degF

TW_GATE_LO, TW_GATE_HI = 10.0, 25.0      # kt

W_MIX   = 0.25
W_TWSPD = 0.45
W_SKY   = 0.15
W_DD    = 0.15

SURFACE_ANCHOR_M = 10.0
M_TO_FT  = 3.28084
MS_TO_KT = 1.94384

_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

def _k_to_f(k):
    return (k - 273.15) * 9.0 / 5.0 + 32.0


# ---------------------------------------------------------------------------
# Field fetch (xarray, one field at a time)
# ---------------------------------------------------------------------------

def _fetch_field(herbie_model, product, search, cycle_dt, fxx, save_dir):
    H = Herbie(cycle_dt, model=herbie_model, product=product,
               fxx=fxx, save_dir=str(save_dir), overwrite=False)
    result = H.xarray(search, remove_grib=False)
    if isinstance(result, list):
        result = result[0] if result else xr.Dataset()
    if isinstance(result, xr.DataArray):
        result = result.to_dataset(name=result.name or "var")
    return result


def _vals(ds):
    for v in ds.data_vars:
        return np.asarray(ds[v].values, dtype=np.float32)
    raise ValueError("Empty dataset — check search string.")

def _fetch_prs_level_safe(fetch, co, prs_product, mb):
    try:
        u_mb = co(_vals(fetch(prs_product, f":UGRD:{mb} mb:")))
        v_mb = co(_vals(fetch(prs_product, f":VGRD:{mb} mb:")))
        h_mb = co(_vals(fetch(prs_product, f":HGT:{mb} mb:")))
        return u_mb, v_mb, h_mb
    except Exception as exc:
        logger.warning("llti: missing %s mb prs fields, skipping level (%s)", mb, exc)
        return None

# ---------------------------------------------------------------------------
# CONUS clip
# ---------------------------------------------------------------------------

def _conus_slices(lat2d, lon2d):
    lon_std = np.where(lon2d > 180.0, lon2d - 360.0, lon2d)
    mask = ((lat2d >= LAT_MIN) & (lat2d <= LAT_MAX) &
            (lon_std >= LON_MIN) & (lon_std <= LON_MAX))
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0 = int(np.where(rows)[0][0]);  r1 = int(np.where(rows)[0][-1])
    c0 = int(np.where(cols)[0][0]);  c1 = int(np.where(cols)[0][-1])
    return slice(r0, r1 + 1), slice(c0, c1 + 1)


# ---------------------------------------------------------------------------
# Thickness-weighted transport wind (vectorized, preserved from original)
# ---------------------------------------------------------------------------

def _compute_transport_wind(u10m, v10m, u_prs, v_prs, hgt_prs, orog, hpbl):
    N, ny, nx = u_prs.shape

    hgt_agl_prs = hgt_prs - orog[np.newaxis, :, :]

    h_anchor = np.full((1, ny, nx), SURFACE_ANCHOR_M, dtype=np.float32)
@@ -219,65 +230,77 @@ def _compute(herbie_model, sfc_product, prs_product, cycle_dt, fxx, step):

    lat2d_full = np.asarray(ds_t2m["latitude"].values,  dtype=np.float32)
    lon2d_full = np.asarray(ds_t2m["longitude"].values, dtype=np.float32)

    rsl, csl = _conus_slices(lat2d_full, lon2d_full)

    def co(arr):
        return arr[rsl, csl][::step, ::step]

    lat2d = co(lat2d_full)
    lon2d = co(np.where(lon2d_full > 180.0, lon2d_full - 360.0, lon2d_full))

    hpbl_m  = co(_vals(ds_hpbl))
    orog_m  = co(_vals(ds_orog))
    u10m    = co(_vals(ds_u10))
    v10m    = co(_vals(ds_v10))
    t2m_k   = co(_vals(ds_t2m))
    dpt_k   = co(_vals(ds_dpt))
    tcc_pct = co(_vals(ds_tcc))

    mix_ft = hpbl_m * M_TO_FT
    t_f    = _k_to_f(t2m_k)
    td_f   = _k_to_f(dpt_k)

    # Pressure-level fields for transport wind
    logger.info("llti: fetching up to %d prs levels (%s)...",
                len(transport_levels), prs_product)
    u_prs_list, v_prs_list, hgt_prs_list = [], [], []
    used_levels = []
    for mb in transport_levels:
        level = _fetch_prs_level_safe(fetch, co, prs_product, mb)
        if level is None:
            continue
        u_mb, v_mb, h_mb = level

        u_prs_list.append(u_mb)
        v_prs_list.append(v_mb)
        hgt_prs_list.append(h_mb)
        used_levels.append(mb)

    logger.info("llti: using transport levels %s", used_levels)
    if u_prs_list:
        u_prs   = np.stack(u_prs_list,   axis=0).astype(np.float32)
        v_prs   = np.stack(v_prs_list,   axis=0).astype(np.float32)
        hgt_prs = np.stack(hgt_prs_list, axis=0).astype(np.float32)
        trspd_kt = _compute_transport_wind(u10m, v10m, u_prs, v_prs,
                                           hgt_prs, orog_m, hpbl_m)
    else:
        logger.warning("llti: no prs levels available; falling back to 10 m wind for transport speed")
        trspd_kt = (np.sqrt(u10m**2 + v10m**2) * MS_TO_KT).astype(np.float32)

    logger.info("llti: computing LLTI...")
    llti2d = _compute_llti(mix_ft, trspd_kt, tcc_pct, t_f, td_f)

    return lat2d, lon2d, llti2d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_llti_arrays(herbie_model: str,
                      sfc_product: str,
                      prs_product: str,
                      cycle_dt: datetime,
                      fxx: int,
                      subsample_step=None):
    """
    Return (lat2d, lon2d, llti2d) for the renderer.
    llti2d: 0-100 (25=low, 50=moderate, 75=high).
    """
    if subsample_step is None:
        subsample_step = DEFAULT_STEP_HRRR if herbie_model == "hrrr" else DEFAULT_STEP_RAP

    cycle = cycle_dt.replace(tzinfo=None)
    cache_key = (herbie_model, sfc_product, prs_product,
                 cycle.isoformat(), fxx, subsample_step)
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and (now - cached["ts"]) < 600:
        return cached["lat"], cached["lon"], cached["llti"]

    lat, lon, llti = _compute(herbie_model, sfc_product, prs_product,
                               cycle, fxx, subsample_step)
    _CACHE[cache_key] = {"ts": now, "lat": lat, "lon": lon, "llti": llti}
    return lat, lon, llti
