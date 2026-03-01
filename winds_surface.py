"""
winds_surface.py  –  HRRR 10-metre wind grid for Colorado streamline animation
================================================================================
Fetches UGRD and VGRD at 10 m above ground from the HRRR sfc product.

Returns a compact grid representation (not a point list) optimised for
client-side particle animation:
    u_flat   flat row-major array of U components (m/s), west = negative
    v_flat   flat row-major array of V components (m/s), south = negative
    lat_min, lat_max, lon_min, lon_max   grid bounds
    rows, cols                           grid dimensions

Download: ~1–2 MB byte-range subset (2 fields from sfc file).
"""

import os
import gc
import time
import pygrib
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from herbie import Herbie

from grib_lock import GRIB_LOCK

# ── Paths ─────────────────────────────────────────────────────────────────────
HERBIE_DIR = Path(os.environ.get("HERBIE_DATA_DIR", "/tmp/herbie"))
HERBIE_DIR.mkdir(parents=True, exist_ok=True)

# ── Colorado clip bounds ───────────────────────────────────────────────────────
CO_LAT_MIN, CO_LAT_MAX = 36.8, 41.2
CO_LON_MIN, CO_LON_MAX = -109.2, -101.9

# ── Download size guard ───────────────────────────────────────────────────────
_MAX_SUBSET_MB = 30

# ── GRIB search string ────────────────────────────────────────────────────────
# Matches exactly 2 messages: UGRD and VGRD at 10 m above ground
SFC_SEARCH = r"(?:UGRD|VGRD):10 m above ground"

# ── Subsampling step ──────────────────────────────────────────────────────────
# HRRR native = 3 km.  step=4 → ~12 km, gives smooth interpolation
# for the particle animation while keeping JSON compact (~2500 grid points).
GRID_STEP = 4

# ── In-memory cache keyed by (cycle_utc, fxx) ────────────────────────────────
_CACHE    = {}
_CLIP_IDX = {}


# ── Herbie helpers ────────────────────────────────────────────────────────────

def _now_utc_hour_naive():
    return datetime.utcnow().replace(minute=0, second=0, microsecond=0)


def _download_subset(cycle: datetime, fxx: int) -> Path:
    H = Herbie(cycle, model="hrrr", product="sfc", fxx=fxx,
               save_dir=str(HERBIE_DIR), overwrite=False)
    result = H.download(searchString=SFC_SEARCH)
    p = Path(result) if result else None
    if p is None or not p.exists():
        raise FileNotFoundError(f"Download failed: sfc 10m wind {cycle} F{fxx:02d}")
    size_mb = p.stat().st_size / 1_000_000
    if size_mb > _MAX_SUBSET_MB:
        raise RuntimeError(
            f"File is {size_mb:.0f} MB — NOMADS returned full file, not byte-range subset. "
            f"Retry when data migrates to AWS (~1–2 hrs after cycle)."
        )
    return p


# ── Grid helpers ──────────────────────────────────────────────────────────────

def _get_clip_idx(lat2d, lon2d):
    shape_key = lat2d.shape
    if shape_key in _CLIP_IDX:
        return _CLIP_IDX[shape_key]
    mask = (
        (lat2d >= CO_LAT_MIN) & (lat2d <= CO_LAT_MAX) &
        (lon2d >= CO_LON_MIN) & (lon2d <= CO_LON_MAX)
    )
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    idx = (r0, r1, c0, c1, GRID_STEP)
    _CLIP_IDX[shape_key] = idx
    return idx


def _clip(data, idx):
    r0, r1, c0, c1, step = idx
    return data[r0:r1, c0:c1][::step, ::step].astype(np.float32)


# ── GRIB reader ───────────────────────────────────────────────────────────────

def _read_uv10(subset_path: Path):
    """
    Read UGRD and VGRD at 10 m above ground.
    Returns (lat_co, lon_co, u10, v10) as clipped Colorado arrays.

    HRRR GRIB names:
        'U component of wind'  typeOfLevel='heightAboveGround'  level=10
        'V component of wind'  typeOfLevel='heightAboveGround'  level=10
    """
    u10 = v10 = None
    lat_co = lon_co = None
    clip_idx = None

    grbs = pygrib.open(str(subset_path))
    for grb in grbs:
        if grb.typeOfLevel != "heightAboveGround" or grb.level != 10:
            continue
        name = grb.name
        # Actual HRRR sfc GRIB names: '10 metre U wind component' / '10 metre V wind component'
        is_u = "10 metre U" in name or name == "U component of wind"
        is_v = "10 metre V" in name or name == "V component of wind"
        if not is_u and not is_v:
            continue

        data, lat2d, lon2d = grb.data()
        lon2d = np.where(lon2d > 180, lon2d - 360, lon2d)

        if clip_idx is None:
            clip_idx = _get_clip_idx(lat2d, lon2d)
            r0, r1, c0, c1, step = clip_idx
            lat_co = lat2d[r0:r1, c0:c1][::step, ::step]
            lon_co = lon2d[r0:r1, c0:c1][::step, ::step]

        clipped = _clip(data, clip_idx)
        if is_u:
            u10 = clipped
        else:
            v10 = clipped

        del data, lat2d, lon2d

    grbs.close()
    gc.collect()

    if u10 is None or v10 is None:
        raise ValueError(
            "Could not find UGRD/VGRD at 10 m above ground. "
            "Check search string against actual GRIB inventory."
        )

    return lat_co, lon_co, u10, v10


# ── Main fetch function ───────────────────────────────────────────────────────

def fetch_surface_wind(cycle_utc: str, fxx: int = 1) -> dict:
    """
    Return a compact U/V grid for client-side streamline animation.

    The response uses a flat array format rather than a point list so the
    JS animation loop can do fast bilinear interpolation:

        u_flat[row * cols + col]  →  U component at that grid cell (m/s)
        v_flat[row * cols + col]  →  V component (m/s)

    Lat/lon run from (lat_min, lon_min) at index [0,0] with uniform
    spacing (lat increases northward row by row).
    """
    cycle = datetime.fromisoformat(
        cycle_utc.replace("Z", "+00:00")
    ).replace(tzinfo=None)
    cycle_aware = cycle.replace(tzinfo=timezone.utc)

    # ── Download under global lock ────────────────────────────────────────────
    if not GRIB_LOCK.acquire(timeout=30):
        raise RuntimeError(
            "GRIB_LOCK timeout — another download is in progress, retry shortly."
        )
    try:
        path = _download_subset(cycle, fxx)
        lat_co, lon_co, u10, v10 = _read_uv10(path)
    finally:
        GRIB_LOCK.release()

    rows, cols = lat_co.shape

    # Grid bounds — used by client to map lat/lon → grid index
    lat_min = float(lat_co.min())
    lat_max = float(lat_co.max())
    lon_min = float(lon_co.min())
    lon_max = float(lon_co.max())

    # Speed, direction, category at each grid point
    spd  = np.sqrt(u10**2 + v10**2) * 1.94384        # m/s → kt
    wdir = (np.degrees(np.arctan2(u10, v10)) + 360) % 360   # met convention
    spd_max = float(np.nanpercentile(spd, 99))

    cat = np.zeros_like(spd, dtype=np.int8)
    cat[spd >=  8] = 1
    cat[spd >= 15] = 2
    cat[spd >= 25] = 3
    cat[spd >= 40] = 4

    dlat = (lat_max - lat_min) / max(rows - 1, 1)
    dlon = (lon_max - lon_min) / max(cols - 1, 1)
    cell_size_deg = round((dlat + dlon) / 2, 4)

    points = []
    for i in range(rows):
        for j in range(cols):
            points.append({
                "lat":  round(float(lat_co[i, j]), 4),
                "lon":  round(float(lon_co[i, j]), 4),
                "spd":  round(float(spd[i, j]),  1),
                "wdir": round(float(wdir[i, j]), 0),
                "cat":  int(cat[i, j]),
            })

    valid_dt  = cycle + timedelta(hours=fxx)
    valid_utc = (valid_dt.replace(tzinfo=timezone.utc)
                 .isoformat(timespec="minutes")
                 .replace("+00:00", "Z"))

    return {
        "points":        points,
        "point_count":   rows * cols,
        "cell_size_deg": cell_size_deg,
        "rows":    rows,
        "cols":    cols,
        "lat_min": round(lat_min, 4),
        "lat_max": round(lat_max, 4),
        "lon_min": round(lon_min, 4),
        "lon_max": round(lon_max, 4),
        "u_flat": [round(float(v), 2) for v in u10.flatten()],
        "v_flat": [round(float(v), 2) for v in v10.flatten()],
        "spd_max_kt": round(spd_max, 1),
        "valid_utc":  valid_utc,
        "cycle_utc":  cycle_aware.isoformat(timespec="minutes").replace("+00:00", "Z"),
        "fxx":        fxx,
    }


def get_surface_wind_cached(cycle_utc: str, fxx: int = 1,
                            ttl_seconds: int = 600) -> dict:
    """Cache keyed by (cycle_utc, fxx)."""
    key    = (cycle_utc, fxx)
    now    = time.time()
    cached = _CACHE.get(key)
    if cached is None or (now - cached["ts"]) > ttl_seconds:
        _CACHE[key] = {"ts": now, "data": fetch_surface_wind(
            cycle_utc=cycle_utc, fxx=fxx)}
    return _CACHE[key]["data"]
