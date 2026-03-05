"""
llti_threat.py
==============
CONUS Low-Level Turbulence Index for model-viewer.

Adapted from llti.py to support:
  - Full CONUS domain (not clipped to Colorado)
  - HRRR: xarray fetch (one field at a time, proven approach)
  - RAP13: pygrib fetch (batch sfc + prs downloads, cfgrib-free)
  - Parameterized transport wind levels (RAP13 skips 875/825 mb)
  - Stripped PNG renderer (model-viewer handles rendering)
  - fetch_llti_arrays() returning numpy 2D arrays for the renderer

Algorithm (preserved exactly from GFE Smart Tool)
  score = W_MIX   * s_mix_eff   (gated by transport wind speed)
        + W_TWSPD * s_twspd
        + W_SKY   * s_sky        (inverted: clear sky -> higher score)
        + W_DD    * s_dd         (dry air -> higher score)
  LLTI  = clip(score, 0, 1) * 100
"""

import gc
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pygrib
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
# RAP13 conservatively skips 875/825 mb
TRANSPORT_LEVELS_HRRR = [950, 925, 900, 875, 850, 825, 800, 750, 700]
TRANSPORT_LEVELS_RAP  = [950, 925, 900, 850, 800, 750, 700]

# RAP13 batch GRIB searches
_RAP_SFC_SEARCH = (
    r":(?:TMP|DPT|HPBL|TCDC):(?:2 m above ground|surface|entire atmosphere):"
    r"|:(?:UGRD|VGRD|HGT):10 m above ground:"
)
_RAP_PRS_LEVELS = "(?:950|925|900|850|800|750|700)"
_RAP_PRS_SEARCH = rf":(?:UGRD|VGRD|HGT):{_RAP_PRS_LEVELS} mb:"

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

_CACHE: dict    = {}
_CLIP_IDX: dict = {}


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

def _k_to_f(k):
    return (k - 273.15) * 9.0 / 5.0 + 32.0

def _k_to_c(k):
    return k - 273.15

def _rh_from_td(T_K, Td_K):
    T_C  = _k_to_c(T_K)
    Td_C = _k_to_c(Td_K)
    e_T  = 6.112 * np.exp(17.67 * T_C  / (T_C  + 243.5))
    e_Td = 6.112 * np.exp(17.67 * Td_C / (Td_C + 243.5))
    return np.clip(100.0 * e_Td / e_T, 0.0, 100.0)


# ---------------------------------------------------------------------------
# CONUS clip helpers
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


def _conus_slices_xr(lat2d, lon2d):
    """For xarray path — returns row/col slices."""
    lon_std = np.where(lon2d > 180.0, lon2d - 360.0, lon2d)
    mask = ((lat2d >= LAT_MIN) & (lat2d <= LAT_MAX) &
            (lon_std >= LON_MIN) & (lon_std <= LON_MAX))
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0 = int(np.where(rows)[0][0]);  r1 = int(np.where(rows)[0][-1])
    c0 = int(np.where(cols)[0][0]);  c1 = int(np.where(cols)[0][-1])
    return slice(r0, r1 + 1), slice(c0, c1 + 1)


# ---------------------------------------------------------------------------
# LLTI algorithm (pure numpy, preserved from GFE)
# ---------------------------------------------------------------------------

def _compute_transport_wind(u10m, v10m, u_prs, v_prs, hgt_prs, orog, hpbl):
    N, ny, nx = u_prs.shape

    hgt_agl_prs = hgt_prs - orog[np.newaxis, :, :]
    h_anchor    = np.full((1, ny, nx), SURFACE_ANCHOR_M, dtype=np.float32)
    h_agl       = np.concatenate([h_anchor, hgt_agl_prs], axis=0)
    u_all       = np.concatenate([u10m[np.newaxis], u_prs], axis=0)
    v_all       = np.concatenate([v10m[np.newaxis], v_prs], axis=0)

    h_below      = np.concatenate([np.zeros((1, ny, nx), dtype=np.float32),
                                    h_agl[:-1]], axis=0)
    midpoint_agl = (h_below + h_agl) / 2.0
    hpbl_3d      = hpbl[np.newaxis, :, :]
    valid        = (h_agl > 0.0) & (midpoint_agl < hpbl_3d)

    h_lower = np.concatenate([np.zeros((1, ny, nx), dtype=np.float32),
                               (h_agl[:-1] + h_agl[1:]) / 2.0], axis=0)
    h_upper = np.concatenate([(h_agl[:-1] + h_agl[1:]) / 2.0,
                               hpbl_3d], axis=0)
    h_lower = np.clip(h_lower, 0.0, hpbl_3d)
    h_upper = np.clip(h_upper, 0.0, hpbl_3d)

    dz        = np.where(valid, np.maximum(h_upper - h_lower, 0.0), 0.0)
    dz_total  = dz.sum(axis=0)
    no_layer  = dz_total < 1.0

    u_mean = (u_all * dz).sum(axis=0) / np.where(no_layer, 1.0, dz_total)
    v_mean = (v_all * dz).sum(axis=0) / np.where(no_layer, 1.0, dz_total)
    u_mean = np.where(no_layer, u10m, u_mean)
    v_mean = np.where(no_layer, v10m, v_mean)

    return (np.sqrt(u_mean**2 + v_mean**2) * MS_TO_KT).astype(np.float32)


def _compute_llti(mix_ft, trspd_kt, sky_pct, t_f, td_f):
    dd      = np.clip(t_f - td_f, 0.0, None)
    s_mix   = np.clip((mix_ft   - MIX_LO)   / max(MIX_HI   - MIX_LO,   1e-6), 0.0, 1.0)
    s_twspd = np.clip((trspd_kt - TWSPD_LO) / max(TWSPD_HI - TWSPD_LO, 1e-6), 0.0, 1.0)
    s_sky   = np.clip((SKY_REF  - sky_pct)  / max(SKY_REF,               1e-6), 0.0, 1.0)
    s_dd    = np.clip((dd       - DD_LO)    / max(DD_HI    - DD_LO,    1e-6), 0.0, 1.0)

    gate      = np.clip((trspd_kt - TW_GATE_LO) /
                        max(TW_GATE_HI - TW_GATE_LO, 1e-6), 0.0, 1.0)
    s_mix_eff = s_mix * gate

    score01 = np.clip(
        W_MIX * s_mix_eff + W_TWSPD * s_twspd +
        W_SKY * s_sky     + W_DD    * s_dd,
        0.0, 1.0,
    )
    return np.nan_to_num((score01 * 100.0).astype(np.float32),
                         nan=0.0, posinf=100.0, neginf=0.0)


# ---------------------------------------------------------------------------
# HRRR path — xarray fetch (one field at a time)
# ---------------------------------------------------------------------------

def _fetch_xr(herbie_model, product, search, cycle_dt, fxx, save_dir):
    H = Herbie(cycle_dt, model=herbie_model, product=product,
               fxx=fxx, save_dir=str(save_dir), overwrite=False)
    result = H.xarray(search, remove_grib=True)
    if isinstance(result, list):
        result = result[0] if result else xr.Dataset()
    if isinstance(result, xr.DataArray):
        result = result.to_dataset(name=result.name or "var")
    return result


def _vals(ds):
    for v in ds.data_vars:
        return np.asarray(ds[v].values, dtype=np.float32)
    raise ValueError("Empty dataset — check search string.")


def _compute_hrrr(cycle_dt, fxx, step):
    transport_levels = TRANSPORT_LEVELS_HRRR

    def save(product):
        d = HERBIE_DIR / f"llti_hrrr_{cycle_dt.strftime('%Y%m%d%H')}_{fxx:02d}_{product}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def fetch(product, search):
        return _fetch_xr("hrrr", product, search, cycle_dt, fxx, save(product))

    logger.info("llti HRRR: fetching sfc fields...")
    ds_t2m  = fetch("sfc", ":TMP:2 m above ground:")
    ds_hpbl = fetch("sfc", ":HPBL:surface:")
    ds_orog = fetch("sfc", ":HGT:surface:")
    ds_u10  = fetch("sfc", ":UGRD:10 m above ground:")
    ds_v10  = fetch("sfc", ":VGRD:10 m above ground:")
    ds_dpt  = fetch("sfc", ":DPT:2 m above ground:")
    ds_tcc  = fetch("sfc", ":TCDC:entire atmosphere:")

    lat2d_full = np.asarray(ds_t2m["latitude"].values,  dtype=np.float32)
    lon2d_full = np.asarray(ds_t2m["longitude"].values, dtype=np.float32)
    rsl, csl   = _conus_slices_xr(lat2d_full, lon2d_full)

    def co(arr):
        return arr[rsl, csl][::step, ::step]

    lat2d   = co(lat2d_full)
    lon2d   = co(np.where(lon2d_full > 180.0, lon2d_full - 360.0, lon2d_full))
    hpbl_m  = co(_vals(ds_hpbl))
    orog_m  = co(_vals(ds_orog))
    u10m    = co(_vals(ds_u10))
    v10m    = co(_vals(ds_v10))
    t2m_k   = co(_vals(ds_t2m))
    dpt_k   = co(_vals(ds_dpt))
    tcc_pct = co(_vals(ds_tcc))

    logger.info("llti HRRR: fetching %d prs levels...", len(transport_levels))
    u_prs_list, v_prs_list, hgt_prs_list = [], [], []
    for mb in transport_levels:
        u_prs_list.append(   co(_vals(fetch("prs", f":UGRD:{mb} mb:"))))
        v_prs_list.append(   co(_vals(fetch("prs", f":VGRD:{mb} mb:"))))
        hgt_prs_list.append( co(_vals(fetch("prs", f":HGT:{mb} mb:"))))

    u_prs   = np.stack(u_prs_list,   axis=0).astype(np.float32)
    v_prs   = np.stack(v_prs_list,   axis=0).astype(np.float32)
    hgt_prs = np.stack(hgt_prs_list, axis=0).astype(np.float32)

    trspd_kt = _compute_transport_wind(u10m, v10m, u_prs, v_prs,
                                        hgt_prs, orog_m, hpbl_m)
    llti2d = _compute_llti(hpbl_m * M_TO_FT, trspd_kt, tcc_pct,
                            _k_to_f(t2m_k), _k_to_f(dpt_k))
    return lat2d, lon2d, llti2d


# ---------------------------------------------------------------------------
# RAP13 path — pygrib batch fetch
# ---------------------------------------------------------------------------

# GRIB name -> field key mapping for sfc fields
_SFC_NAME_MAP = {
    "Temperature":              "T2m",
    "Dew point temperature":    "Td2m",
    "Planetary boundary layer height": "HPBL",
    "Geopotential Height":      "OROG",   # HGT:surface
    "U component of wind":      "U10",
    "V component of wind":      "V10",
    "Total Cloud Cover":        "TCC",
    "Total cloud cover":        "TCC",
}

# For prs fields
_PRS_U_NAMES   = {"U component of wind"}
_PRS_V_NAMES   = {"V component of wind"}
_PRS_HGT_NAMES = {"Geopotential Height", "Geopotential height"}


def _read_rap_sfc(path, step):
    """Read all needed sfc fields from a single pygrib file."""
    out = {}
    lat_out = lon_out = clip_idx = None

    grbs = pygrib.open(str(path))
    try:
        for grb in grbs:
            key = _SFC_NAME_MAP.get(grb.name)
            if key is None:
                continue
            # Disambiguate T2m vs surface HGT vs 10m winds by typeOfLevel
            if grb.name == "Temperature" and grb.typeOfLevel != "heightAboveGround":
                continue
            if grb.name in ("U component of wind", "V component of wind"):
                if grb.typeOfLevel != "heightAboveGround" or grb.level != 10:
                    continue
            if grb.name in ("Geopotential Height", "Geopotential height"):
                if grb.typeOfLevel != "surface":
                    continue
            if key in out:
                continue  # already have this field

            if clip_idx is None:
                data, lat2d, lon2d = grb.data()
                lon2d    = np.where(lon2d > 180, lon2d - 360, lon2d)
                clip_idx = _get_clip_idx(lat2d, lon2d)
                r0, r1, c0, c1 = clip_idx
                lat_out  = lat2d[r0:r1, c0:c1][::step, ::step].astype(np.float32)
                lon_out  = lon2d[r0:r1, c0:c1][::step, ::step].astype(np.float32)
                del lat2d, lon2d
            else:
                data = grb.values

            out[key] = _clip(data, clip_idx, step)
            del data
    finally:
        grbs.close()
        gc.collect()

    required = ["T2m", "Td2m", "HPBL", "OROG", "U10", "V10", "TCC"]
    missing  = [k for k in required if k not in out]
    if missing:
        raise ValueError(f"llti RAP13 sfc: missing fields {missing}")

    return lat_out, lon_out, clip_idx, out


def _read_rap_prs(path, transport_levels, clip_idx, step):
    """Read U, V, HGT at each transport level from a single pygrib file."""
    levels_set = frozenset(transport_levels)
    U   = {}
    V   = {}
    HGT = {}

    grbs = pygrib.open(str(path))
    try:
        for grb in grbs:
            if grb.typeOfLevel != "isobaricInhPa":
                continue
            lev = grb.level
            if lev not in levels_set:
                continue
            if grb.name in _PRS_U_NAMES and lev not in U:
                U[lev]   = _clip(grb.values, clip_idx, step)
            elif grb.name in _PRS_V_NAMES and lev not in V:
                V[lev]   = _clip(grb.values, clip_idx, step)
            elif grb.name in _PRS_HGT_NAMES and lev not in HGT:
                HGT[lev] = _clip(grb.values, clip_idx, step)
    finally:
        grbs.close()
        gc.collect()

    missing = [l for l in transport_levels
               if l not in U or l not in V or l not in HGT]
    if missing:
        logger.warning("llti RAP13 prs: missing U/V/HGT at levels %s — skipping.", missing)

    usable = sorted([l for l in transport_levels
                     if l in U and l in V and l in HGT])
    if not usable:
        raise ValueError("llti RAP13 prs: no usable levels found.")

    return (
        np.stack([U[l]   for l in usable], axis=0).astype(np.float32),
        np.stack([V[l]   for l in usable], axis=0).astype(np.float32),
        np.stack([HGT[l] for l in usable], axis=0).astype(np.float32),
    )


def _compute_rap(sfc_product, prs_product, cycle_dt, fxx, step):
    transport_levels = TRANSPORT_LEVELS_RAP

    logger.info("llti RAP13: downloading sfc fields (%s)...", sfc_product)
    H_sfc = Herbie(cycle_dt, model="rap", product=sfc_product,
                   fxx=fxx, save_dir=str(HERBIE_DIR), overwrite=False)
    result_sfc = H_sfc.download(searchString=_RAP_SFC_SEARCH)
    sfc_path = Path(result_sfc) if result_sfc else None
    if sfc_path is None or not sfc_path.exists():
        raise FileNotFoundError(f"llti RAP13: sfc download failed for {sfc_product} F{fxx:02d}")

    logger.info("llti RAP13: downloading prs fields (%s)...", prs_product)
    H_prs = Herbie(cycle_dt, model="rap", product=prs_product,
                   fxx=fxx, save_dir=str(HERBIE_DIR), overwrite=False)
    result_prs = H_prs.download(searchString=_RAP_PRS_SEARCH)
    prs_path = Path(result_prs) if result_prs else None
    if prs_path is None or not prs_path.exists():
        raise FileNotFoundError(f"llti RAP13: prs download failed for {prs_product} F{fxx:02d}")
  
    logger.info("llti RAP13: reading sfc fields...")
    lat2d, lon2d, clip_idx, sfc = _read_rap_sfc(sfc_path, step)

    logger.info("llti RAP13: reading prs fields...")
    u_prs, v_prs, hgt_prs = _read_rap_prs(prs_path, transport_levels,
                                            clip_idx, step)

    hpbl_m  = sfc["HPBL"]
    orog_m  = sfc["OROG"]
    u10m    = sfc["U10"]
    v10m    = sfc["V10"]
    t2m_k   = sfc["T2m"]
    dpt_k   = sfc["Td2m"]
    tcc_pct = sfc["TCC"]

    logger.info("llti RAP13: computing transport wind and LLTI...")
    trspd_kt = _compute_transport_wind(u10m, v10m, u_prs, v_prs,
                                        hgt_prs, orog_m, hpbl_m)
    llti2d = _compute_llti(hpbl_m * M_TO_FT, trspd_kt, tcc_pct,
                            _k_to_f(t2m_k), _k_to_f(dpt_k))
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
    HRRR uses xarray fetch; RAP13 uses pygrib batch fetch.
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

    if herbie_model == "hrrr":
        lat, lon, llti = _compute_hrrr(cycle, fxx, subsample_step)
    else:
        lat, lon, llti = _compute_rap(sfc_product, prs_product,
                                       cycle, fxx, subsample_step)

    _CACHE[cache_key] = {"ts": now, "lat": lat, "lon": lon, "llti": llti}
    return lat, lon, llti
