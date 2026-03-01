"""
winds.py - HRRR Wind Gust fetcher for Colorado
Uses pygrib for synchronous GRIB2 reading (no lazy loading race conditions).

Field confirmed via /debug/grib_fields:
  name="Wind speed (gust)", typeOfLevel="surface", level=0
"""

import os
import time
import pygrib
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from herbie import Herbie

HERBIE_DIR = Path(os.environ.get("HERBIE_DATA_DIR", "/tmp/herbie"))
HERBIE_DIR.mkdir(parents=True, exist_ok=True)

CO_LAT_MIN = 36.8
CO_LAT_MAX = 41.2
CO_LON_MIN = -109.2
CO_LON_MAX = -101.9

MAX_FXX    = 12   # slider goes F01-F12

_CACHE        = {}   # keyed by (cycle_str, fxx)
_STATUS_CACHE = {"ts": 0, "data": None}


def _now_utc_hour_naive():
    return datetime.utcnow().replace(minute=0, second=0, microsecond=0)


def _find_latest_hrrr_cycle(max_lookback_hours=6):
    base = _now_utc_hour_naive()
    for h in range(max_lookback_hours + 1):
        candidate = base - timedelta(hours=h)
        try:
            H = Herbie(candidate, model="hrrr", product="sfc", fxx=0,
                       save_dir=str(HERBIE_DIR), overwrite=False)
            H.inventory()
            return candidate
        except Exception:
            continue
    return base - timedelta(hours=2)


def _check_fxx_available(cycle: datetime, fxx: int) -> bool:
    """Fast availability check — only fetches the tiny .idx file, not the GRIB."""
    try:
        H = Herbie(cycle, model="hrrr", product="sfc", fxx=fxx,
                   save_dir=str(HERBIE_DIR), overwrite=False)
        H.inventory()
        return True
    except Exception:
        return False


def get_cycle_status() -> dict:
    """
    Check which forecast hours are available for the latest TWO HRRR cycles.
    Uses a thread pool so all 24 inventory checks run in parallel (~2-3s total).
    Returns a list of cycle dicts with available_hours and pct_complete.
    """
    latest = _find_latest_hrrr_cycle()
    cycles = [latest, latest - timedelta(hours=1)]

    results = []
    for cycle in cycles:
        cycle_aware = cycle.replace(tzinfo=timezone.utc)
        fxx_range   = list(range(1, MAX_FXX + 1))

        # Parallel availability checks
        available = {}
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(_check_fxx_available, cycle, fxx): fxx
                       for fxx in fxx_range}
            for future in as_completed(futures):
                fxx = futures[future]
                try:
                    available[fxx] = future.result()
                except Exception:
                    available[fxx] = False

        avail_hours = sorted(fxx for fxx, ok in available.items() if ok)
        pct         = round(len(avail_hours) / MAX_FXX * 100)

        results.append({
            "cycle_utc":       cycle_aware.isoformat(timespec="minutes").replace("+00:00", "Z"),
            "available_hours": avail_hours,
            "total_hours":     MAX_FXX,
            "pct_complete":    pct,
        })

    return {"cycles": results, "checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def get_cycle_status_cached(ttl_seconds: int = 300) -> dict:
    now = time.time()
    if _STATUS_CACHE["data"] is None or (now - _STATUS_CACHE["ts"]) > ttl_seconds:
        _STATUS_CACHE["data"] = get_cycle_status()
        _STATUS_CACHE["ts"]   = now
    return _STATUS_CACHE["data"]


def fetch_hrrr_gusts(cycle_utc: str, fxx: int = 1) -> dict:
    """
    Fetch HRRR surface wind gusts for a specific cycle + forecast hour.
    cycle_utc is an ISO string like '2026-02-22T01:00Z'.
    """
    # Parse cycle string back to naive UTC datetime
    cycle = datetime.fromisoformat(cycle_utc.replace("Z", "+00:00")).replace(tzinfo=None)
    cycle_aware = cycle.replace(tzinfo=timezone.utc)

    H = Herbie(cycle, model="hrrr", product="sfc", fxx=fxx,
               save_dir=str(HERBIE_DIR), overwrite=False)
    grib_path = Path(H.download())

    if not grib_path.exists():
        raise FileNotFoundError(f"GRIB2 file not found after download: {grib_path}")

    grbs = pygrib.open(str(grib_path))
    try:
        msgs = grbs.select(name="Wind speed (gust)", typeOfLevel="surface", level=0)
    except ValueError:
        grbs.close()
        raise ValueError("Could not find 'Wind speed (gust)' at surface/level=0.")

    gust_arr, lat2d, lon2d = msgs[0].data()
    grbs.close()

    lon2d = np.where(lon2d > 180, lon2d - 360, lon2d)

    raw_max = float(np.nanmax(gust_arr))
    raw_min = float(np.nanmin(gust_arr))
    if raw_max > 150 or raw_min < 0:
        raise ValueError(
            f"Gust values out of physical range "
            f"(min={raw_min:.1f}, max={raw_max:.1f} m/s). Wrong GRIB field."
        )

    mask = (
        (lat2d >= CO_LAT_MIN) & (lat2d <= CO_LAT_MAX) &
        (lon2d >= CO_LON_MIN) & (lon2d <= CO_LON_MAX)
    )
    rows, cols = np.where(mask)
    if len(rows) == 0:
        raise ValueError("No HRRR grid points found inside Colorado bounding box.")

    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1

    step    = 2
    lat_ds  = lat2d[r0:r1, c0:c1][::step, ::step]
    lon_ds  = lon2d[r0:r1, c0:c1][::step, ::step]
    gust_ds = gust_arr[r0:r1, c0:c1][::step, ::step] * 1.94384  # m/s -> knots

    points = []
    for i in range(lat_ds.shape[0]):
        for j in range(lat_ds.shape[1]):
            g = float(gust_ds[i, j])
            if np.isnan(g):
                continue
            points.append({
                "lat":     round(float(lat_ds[i, j]), 4),
                "lon":     round(float(lon_ds[i, j]), 4),
                "gust_kt": round(g, 1),
            })

    valid_dt = (cycle + timedelta(hours=fxx)).replace(tzinfo=timezone.utc)
    return {
        "model":         "HRRR",
        "cycle_utc":     cycle_aware.isoformat(timespec="minutes").replace("+00:00", "Z"),
        "valid_utc":     valid_dt.isoformat(timespec="minutes").replace("+00:00", "Z"),
        "fxx":           fxx,
        "cell_size_deg": 0.055,
        "point_count":   len(points),
        "points":        points,
    }


def get_hrrr_gusts_cached(cycle_utc: str, fxx: int = 1, ttl_seconds: int = 600) -> dict:
    """Cache keyed by (cycle_utc, fxx) so every combination is stored independently."""
    key    = (cycle_utc, fxx)
    now    = time.time()
    cached = _CACHE.get(key)
    if cached is None or (now - cached["ts"]) > ttl_seconds:
        _CACHE[key] = {"ts": now, "data": fetch_hrrr_gusts(cycle_utc=cycle_utc, fxx=fxx)}
    return _CACHE[key]["data"]
