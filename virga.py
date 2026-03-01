"""
virga.py  –  HRRR-based Virga Potential calculator for Colorado
===============================================================
Key optimisation: grb.data() is called exactly ONCE (for lat/lon).
All subsequent messages use grb.values which skips the expensive
pyproj Lambert Conformal unprojection entirely.

Science
-------
1. Upper saturated layer (700-500 mb): mean RH >= 80% over 200 mb depth
2. Max 100 mb RH decrease in column (850-500 mb): evaporation zone
3. Cloud base wind at level of max decrease (kt): virga shaft momentum
RH from T + Td via August-Roche-Magnus approximation.
"""

import os
import time
import pygrib
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from herbie import Herbie

HERBIE_DIR = Path(os.environ.get("HERBIE_DATA_DIR", "/tmp/herbie"))
HERBIE_DIR.mkdir(parents=True, exist_ok=True)

CO_LAT_MIN, CO_LAT_MAX = 36.8, 41.2
CO_LON_MIN, CO_LON_MAX = -109.2, -101.9

LEVELS_MB  = [500, 550, 600, 650, 700, 750, 800, 850]
LEVELS_SET = frozenset(LEVELS_MB)

# Herbie searchString — matches our 4 variables at our 15 specific levels only.
# Written without capture groups to avoid pandas UserWarning.
# IDX lines look like: "TMP:500 mb:1 hour fcst"
# Non-capturing groups avoid pandas UserWarning and correctly scope the alternation.
# Without (?:...) the | operator would match ANY line containing "550 mb" etc.
SEARCH_STRING = r"(?:TMP|DPT|UGRD|VGRD):(?:500|550|600|650|700|750|800|850) mb"

_CACHE    = {}
_CLIP_IDX = {}   # cache (r0,r1,c0,c1,step) by grid shape

# Use the global GRIB lock shared with prefetch/froude/winds
# so background prefetch and user requests never compete for memory.
from grib_lock import GRIB_LOCK as _DOWNLOAD_LOCK


# ── Herbie helpers ────────────────────────────────────────────────────────────

def _now_utc_hour_naive():
    return datetime.utcnow().replace(minute=0, second=0, microsecond=0)


_MAX_SUBSET_MB = 50

def _download_subset(cycle, fxx):
    """
    Download only TMP/DPT/UGRD/VGRD messages from the prs file.
    Raises RuntimeError if file exceeds _MAX_SUBSET_MB — means NOMADS
    returned the full file (no byte-range support).
    """
    H = Herbie(cycle, model="hrrr", product="prs", fxx=fxx,
               save_dir=str(HERBIE_DIR), overwrite=False)
    result = H.download(searchString=SEARCH_STRING)
    p = Path(result) if result else None
    if p is None or not p.exists():
        raise FileNotFoundError(f"Download failed for prs {cycle} F{fxx:02d}")
    size_mb = p.stat().st_size / 1_000_000
    if size_mb > _MAX_SUBSET_MB:
        raise RuntimeError(
            f"Downloaded file is {size_mb:.0f} MB — NOMADS returned full file "
            f"(no byte-range support). Try again when data moves to AWS (~1-2 hrs)."
        )
    return p


# ── Clip helpers ──────────────────────────────────────────────────────────────

def _get_clip_idx(lat2d, lon2d, step=2):
    key = lat2d.shape
    if key not in _CLIP_IDX:
        mask = (
            (lat2d >= CO_LAT_MIN) & (lat2d <= CO_LAT_MAX) &
            (lon2d >= CO_LON_MIN) & (lon2d <= CO_LON_MAX)
        )
        rows, cols = np.where(mask)
        if len(rows) == 0:
            raise ValueError("No HRRR grid points inside Colorado bounding box.")
        _CLIP_IDX[key] = (rows.min(), rows.max() + 1,
                          cols.min(), cols.max() + 1, step)
    return _CLIP_IDX[key]


def _clip(arr, idx):
    r0, r1, c0, c1, step = idx
    return arr[r0:r1, c0:c1][::step, ::step]


# ── Physics ───────────────────────────────────────────────────────────────────

def _rh(T_K, Td_K):
    T_C  = T_K  - 273.15
    Td_C = Td_K - 273.15
    e_T  = 6.112 * np.exp(17.67 * T_C  / (T_C  + 243.5))
    e_Td = 6.112 * np.exp(17.67 * Td_C / (Td_C + 243.5))
    return np.clip(100.0 * e_Td / e_T, 0.0, 100.0)


def _virga_category(pct):
    cat = np.zeros_like(pct, dtype=int)   # 0 = negligible (<20%)
    cat[pct >= 20] = 1
    cat[pct >= 40] = 2
    cat[pct >= 60] = 3
    cat[pct >= 80] = 4
    return cat


# ── Single-pass reader — lat/lon computed exactly once ────────────────────────

def _read_subset_clipped(subset_path):
    """
    Read the prs subset file in a single pass.

    Critical memory optimisation:
      - First qualifying message  → grb.data()  to get data + lat/lon arrays
      - All subsequent messages   → grb.values   (no pyproj call, no lat/lon alloc)

    This reduces pyproj Lambert Conformal allocations from N×2×8MB to 1×2×8MB.
    """
    T_co  = {}
    Td_co = {}
    U_co  = {}
    V_co  = {}
    lat_co = lon_co = None
    idx    = None

    name_map = {
        "Temperature":           "T",
        "Dew point temperature": "Td",
        "U component of wind":   "U",
        "V component of wind":   "V",
    }

    grbs = pygrib.open(str(subset_path))
    for grb in grbs:
        if grb.typeOfLevel != "isobaricInhPa":
            continue
        lev = grb.level
        if lev not in LEVELS_SET:
            continue
        key = name_map.get(grb.name)
        if key is None:
            continue

        if idx is None:
            # First match: pay the pyproj cost once to get lat/lon
            data, lat2d, lon2d = grb.data()
            lon2d = np.where(lon2d > 180, lon2d - 360, lon2d)
            idx    = _get_clip_idx(lat2d, lon2d)
            r0, r1, c0, c1, step = idx
            lat_co = lat2d[r0:r1, c0:c1][::step, ::step]
            lon_co = lon2d[r0:r1, c0:c1][::step, ::step]
            del lat2d, lon2d
        else:
            # All subsequent messages: values only, no lat/lon allocation
            data = grb.values

        small = _clip(data, idx)
        del data

        if   key == "T":  T_co[lev]  = small
        elif key == "Td": Td_co[lev] = small
        elif key == "U":  U_co[lev]  = small
        elif key == "V":  V_co[lev]  = small

    grbs.close()

    missing = [l for l in LEVELS_MB if l not in T_co or l not in Td_co]
    if missing:
        raise ValueError(f"Missing T/Td at levels: {missing} in {subset_path.name}")

    return lat_co, lon_co, T_co, Td_co, U_co, V_co


# ── Main fetch ────────────────────────────────────────────────────────────────

def fetch_virga(cycle_utc: str, fxx: int = 1) -> dict:
    cycle = datetime.fromisoformat(
        cycle_utc.replace("Z", "+00:00")
    ).replace(tzinfo=None)
    cycle_aware = cycle.replace(tzinfo=timezone.utc)

    if not _DOWNLOAD_LOCK.acquire(timeout=30):
        raise RuntimeError("GRIB_LOCK timeout — another download is in progress, retry in a moment.")
    try:
        subset_path = _download_subset(cycle, fxx)
        lat_co, lon_co, T_co, Td_co, U_co, V_co = _read_subset_clipped(subset_path)
    finally:
        _DOWNLOAD_LOCK.release()
    shape = lat_co.shape

    import gc
    rh_co = {lev: _rh(T_co[lev], Td_co[lev]) for lev in LEVELS_MB}
    del T_co, Td_co
    gc.collect()   # explicitly free GRIB arrays before column analysis

    # ── 1. Upper saturated layer (700-500 mb) ─────────────────────────────────
    upper_levels = [l for l in LEVELS_MB if l <= 700]
    max_upper_rh = np.zeros(shape)

    for lev_top in upper_levels:
        window = [l for l in upper_levels if lev_top <= l <= lev_top + 200]
        if len(window) < 2:
            continue
        mean_rh = np.mean([rh_co[l] for l in window], axis=0)
        max_upper_rh = np.maximum(max_upper_rh, mean_rh)

    upper_cloud = max_upper_rh >= 80.0

    # ── 2. Max 100 mb RH decrease in column (850→500 mb) ─────────────────────
    max_rh_decrease    = np.zeros(shape)
    cloud_base_wind_kt = np.zeros(shape)

    for lev_bot in sorted(LEVELS_MB, reverse=True):
        lev_top = lev_bot - 100
        if lev_top not in rh_co:
            continue
        decrease_here = rh_co[lev_bot] - rh_co[lev_top]

        wind_lev = min(U_co.keys(), key=lambda l: abs(l - (lev_bot - 50)))
        wspd_kt  = np.sqrt(U_co[wind_lev]**2 + V_co[wind_lev]**2) * 1.94384

        better             = decrease_here > max_rh_decrease
        max_rh_decrease    = np.where(better, decrease_here,    max_rh_decrease)
        cloud_base_wind_kt = np.where(better, wspd_kt,          cloud_base_wind_kt)

    # ── 3. Mask and categorise ────────────────────────────────────────────────
    virga_pct = np.where(upper_cloud, np.clip(max_rh_decrease, 0, 100), 0.0)
    cat       = _virga_category(virga_pct)

    points = []
    ny, nx = shape
    for i in range(ny):
        for j in range(nx):
            vpct = float(virga_pct[i, j])
            points.append({
                "lat":        round(float(lat_co[i, j]), 4),
                "lon":        round(float(lon_co[i, j]), 4),
                "virga_pct":  round(vpct, 1),
                "cat":        int(cat[i, j]),
                "cb_wind_kt": round(float(cloud_base_wind_kt[i, j]), 1),
                "upper_rh":   round(float(max_upper_rh[i, j]), 1),
            })

    valid_dt = (cycle + timedelta(hours=fxx)).replace(tzinfo=timezone.utc)
    return {
        "model":         "HRRR",
        "product":       "prs (subset)",
        "cycle_utc":     cycle_aware.isoformat(timespec="minutes").replace("+00:00", "Z"),
        "valid_utc":     valid_dt.isoformat(timespec="minutes").replace("+00:00", "Z"),
        "fxx":           fxx,
        "cell_size_deg": 0.055,
        "point_count":   len(points),
        "points":        points,
    }


# ── Cache wrapper ─────────────────────────────────────────────────────────────

def get_virga_cached(cycle_utc: str, fxx: int = 1, ttl_seconds: int = 600) -> dict:
    key    = (cycle_utc, fxx)
    now    = time.time()
    cached = _CACHE.get(key)
    if cached is None or (now - cached["ts"]) > ttl_seconds:
        _CACHE[key] = {"ts": now, "data": fetch_virga(cycle_utc=cycle_utc, fxx=fxx)}
    return _CACHE[key]["data"]
