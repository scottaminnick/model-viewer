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
TRANSPORT_LEVELS_RAP  = [950, 925, 900, 850, 800, 750, 700]

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
    result = H.xarray(search, remove_grib=True)   # ← was False
    if isinstance(result, list):
        result = result[0] if result else xr.Dataset()
    if isinstance(result, xr.DataArray):
        result = result.to_dataset(name=result.name or "var")
    return result

def _vals(ds):
    for v in ds.data_vars:
        return np.asarray(ds[v].values, dtype=np.float32)
    raise ValueError("Empty dataset — check search string.")


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
    h_agl    = np.concatenate([h_anchor, hgt_agl_prs], axis=0)

    u_all = np.concatenate([u10m[np.newaxis], u_prs], axis=0)
    v_all = np.concatenate([v10m[np.newaxis], v_prs], axis=0)

    h_below      = np.concatenate([np.zeros((1, ny, nx), dtype=np.float32),
                                    h_agl[:-1]], axis=0)
    midpoint_agl = (h_below + h_agl) / 2.0

    hpbl_3d = hpbl[np.newaxis, :, :]
    valid   = (h_agl > 0.0) & (midpoint_agl < hpbl_3d)

    h_lower = np.concatenate([np.zeros((1, ny, nx), dtype=np.float32),
                               (h_agl[:-1] + h_agl[1:]) / 2.0], axis=0)
    h_upper = np.concatenate([(h_agl[:-1] + h_agl[1:]) / 2.0,
                               hpbl_3d], axis=0)

    h_lower = np.clip(h_lower, 0.0, hpbl_3d)
    h_upper = np.clip(h_upper, 0.0, hpbl_3d)

    dz       = np.where(valid, np.maximum(h_upper - h_lower, 0.0), 0.0)
    dz_total = dz.sum(axis=0)
    no_layer = dz_total < 1.0

    u_mean = (u_all * dz).sum(axis=0) / np.where(no_layer, 1.0, dz_total)
    v_mean = (v_all * dz).sum(axis=0) / np.where(no_layer, 1.0, dz_total)

    u_mean = np.where(no_layer, u10m, u_mean)
    v_mean = np.where(no_layer, v10m, v_mean)

    return (np.sqrt(u_mean**2 + v_mean**2) * MS_TO_KT).astype(np.float32)


# ---------------------------------------------------------------------------
# LLTI algorithm (pure numpy, preserved from GFE)
# ---------------------------------------------------------------------------

def _normalize(a, lo, hi):
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def _compute_llti(mix_ft, trspd_kt, sky_pct, t_f, td_f):
    dd = np.clip(t_f - td_f, 0.0, None)

    s_mix   = _normalize(mix_ft,   MIX_LO,   MIX_HI)
    s_twspd = _normalize(trspd_kt, TWSPD_LO, TWSPD_HI)
    s_sky   = np.clip((SKY_REF - sky_pct) / max(SKY_REF, 1e-6), 0.0, 1.0)
    s_dd    = _normalize(dd, DD_LO, DD_HI)

    gate      = np.clip((trspd_kt - TW_GATE_LO) /
                        max(TW_GATE_HI - TW_GATE_LO, 1e-6), 0.0, 1.0)
    s_mix_eff = s_mix * gate

    score01 = np.clip(
        W_MIX * s_mix_eff + W_TWSPD * s_twspd +
        W_SKY * s_sky     + W_DD    * s_dd,
        0.0, 1.0,
    )
    out = (score01 * 100.0).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=100.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Core fetch + compute
# ---------------------------------------------------------------------------

def _compute(herbie_model, sfc_product, prs_product, cycle_dt, fxx, step):
    transport_levels = (TRANSPORT_LEVELS_HRRR if herbie_model == "hrrr"
                        else TRANSPORT_LEVELS_RAP)

    # One save_dir per product to avoid Herbie hash collisions
    def save(product):
        d = HERBIE_DIR / f"llti_{herbie_model}_{cycle_dt.strftime('%Y%m%d%H')}_{fxx:02d}_{product}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def fetch(product, search):
        return _fetch_field(herbie_model, product, search,
                            cycle_dt, fxx, save(product))

    # Surface fields
    logger.info("llti: fetching sfc fields (%s)...", sfc_product)
    ds_t2m  = fetch(sfc_product, ":TMP:2 m above ground:")
    ds_hpbl = fetch(sfc_product, ":HPBL:surface:")
    ds_orog = fetch(sfc_product, ":HGT:surface:")
    ds_u10  = fetch(sfc_product, ":UGRD:10 m above ground:")
    ds_v10  = fetch(sfc_product, ":VGRD:10 m above ground:")
    ds_dpt  = fetch(sfc_product, ":DPT:2 m above ground:")
    ds_tcc  = fetch(sfc_product, ":TCDC:entire atmosphere:")

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
    logger.info("llti: fetching %d prs levels (%s)...",
                len(transport_levels), prs_product)
    u_prs_list, v_prs_list, hgt_prs_list = [], [], []
    for mb in transport_levels:
        u_prs_list.append(   co(_vals(fetch(prs_product, f":UGRD:{mb} mb:"))))
        v_prs_list.append(   co(_vals(fetch(prs_product, f":VGRD:{mb} mb:"))))
        hgt_prs_list.append( co(_vals(fetch(prs_product, f":HGT:{mb} mb:"))))

    u_prs   = np.stack(u_prs_list,   axis=0).astype(np.float32)
    v_prs   = np.stack(v_prs_list,   axis=0).astype(np.float32)
    hgt_prs = np.stack(hgt_prs_list, axis=0).astype(np.float32)

    logger.info("llti: computing transport wind and LLTI...")
    trspd_kt = _compute_transport_wind(u10m, v10m, u_prs, v_prs,
                                        hgt_prs, orog_m, hpbl_m)
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
