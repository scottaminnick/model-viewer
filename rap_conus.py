"""
rap_conus.py  –  RAP13 CONUS wind gust visualization
Fetches GUST:surface from the latest RAP cycle and returns a
point list suitable for the Leaflet map.

RAP runs hourly, forecasts F01–F18.
Grid: ~13 km CONUS (~1100 x 700 native points).
Stride=5 → ~52 km spacing → ~30,000 points rendered.
"""

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from herbie import Herbie

HERBIE_DIR = Path(os.environ.get("HERBIE_DATA_DIR", "/tmp/herbie"))
HERBIE_DIR.mkdir(parents=True, exist_ok=True)

# CONUS bounding box
LAT_MIN, LAT_MAX =  22.0,  52.0
LON_MIN, LON_MAX = -126.0, -64.0

# Grid stride — every Nth point
_STRIDE = 5

# ── in-memory cache keyed by (cycle_utc, fxx) ────────────────────────────────
_CACHE: dict = {}
_STATUS_CACHE: dict = {"ts": 0, "data": None}


# ── cycle helpers ─────────────────────────────────────────────────────────────
def _utc_floor_hour() -> datetime:
    """Current UTC time floored to the hour, timezone-naive (Herbie convention)."""
    now = datetime.utcnow()
    return now.replace(minute=0, second=0, microsecond=0)


def _find_latest_rap_cycle(max_lookback: int = 6) -> datetime:
    """
    Walk backwards from the current hour until we find a RAP cycle
    that has an F01 file on AWS.
    """
    base = _utc_floor_hour()
    for h in range(max_lookback + 1):
        candidate = base - timedelta(hours=h)
        try:
            H = Herbie(
                candidate, model="rap", product="wrfmsl", fxx=1,
                save_dir=str(HERBIE_DIR / "rap_conus_probe"),
                overwrite=False,
            )
            H.inventory()          # raises if file not on AWS yet
            return candidate
        except Exception:
            continue
    return base - timedelta(hours=1)   # safe fallback


def get_rap_cycle_status_cached(ttl_seconds: int = 300) -> dict:
    """
    Return a status dict for the latest 3 RAP cycles showing which
    forecast hours are available.  Cached for ttl_seconds.
    """
    now = time.time()
    if _STATUS_CACHE["data"] and (now - _STATUS_CACHE["ts"]) < ttl_seconds:
        return _STATUS_CACHE["data"]

    latest = _find_latest_rap_cycle()
    cycles = []

    for offset in range(3):          # latest, -1h, -2h
        cycle_dt = latest - timedelta(hours=offset)
        cycle_str = cycle_dt.replace(tzinfo=timezone.utc).isoformat(timespec="minutes")

        available = []
        for fxx in range(1, 19):     # RAP goes to F18
            try:
                H = Herbie(
                    cycle_dt, model="rap", product="wrfmsl", fxx=fxx,
                    save_dir=str(HERBIE_DIR / "rap_conus_probe"),
                    overwrite=False,
                )
                H.inventory()
                available.append(fxx)
            except Exception:
                break               # once one hour is missing, later ones won't exist

        pct = round(len(available) / 18 * 100)
        cycles.append({
            "cycle_utc":       cycle_str,
            "available_hours": available,
            "pct_complete":    pct,
            "cached_hours":    {
                "gusts": [
                    fxx for fxx in available
                    if (cycle_str, fxx) in _CACHE
                ]
            },
        })

    result = {"model": "RAP", "cycles": cycles}
    _STATUS_CACHE["data"] = result
    _STATUS_CACHE["ts"]   = now
    return result


# ── GRIB fetch ────────────────────────────────────────────────────────────────
def _as_dataset(obj) -> xr.Dataset:
    """Normalize Herbie.xarray() return value to xr.Dataset."""
    if isinstance(obj, xr.Dataset):
        return obj
    if isinstance(obj, list):
        if not obj:
            raise ValueError("Herbie returned empty list.")
        if all(isinstance(x, xr.Dataset) for x in obj):
            return xr.merge(obj, compat="override", combine_attrs="override")
        if all(isinstance(x, xr.DataArray) for x in obj):
            return xr.merge(
                [da.to_dataset(name=da.name or f"var_{i}") for i, da in enumerate(obj)],
                compat="override", combine_attrs="override",
            )
        return _as_dataset(obj[0])
    raise TypeError(f"Unexpected Herbie return type: {type(obj)}")


def fetch_rap_conus_gusts(cycle_utc: str, fxx: int) -> dict:
    """
    Fetch RAP13 GUST:surface for the full CONUS domain.

    Parameters
    ----------
    cycle_utc : ISO string, e.g. "2026-03-01T12:00+00:00"
    fxx       : forecast hour 1–18

    Returns
    -------
    dict with 'points' list and metadata fields expected by the map JS.
    """
    # Parse cycle string → naive datetime for Herbie
    cycle_dt = datetime.fromisoformat(cycle_utc).replace(tzinfo=None)
    valid_dt  = (cycle_dt + timedelta(hours=fxx)).replace(tzinfo=timezone.utc)

    # Per-product directory keeps byte-range hashes from colliding
    save_dir = HERBIE_DIR / f"rap_conus_{cycle_dt.strftime('%Y%m%d%H')}_{fxx:02d}"
    save_dir.mkdir(parents=True, exist_ok=True)

    H = Herbie(
        cycle_dt, model="rap", product="wrfmsl", fxx=fxx,
        save_dir=str(save_dir), overwrite=False,
    )

    # Try several searchstrings — RAP product naming is less consistent than HRRR
    gust_ds = None
    for search in [":GUST:surface:", ":10u:10 m above ground:", ":WIND:10 m above"]:
        try:
            raw = H.xarray(search, remove_grib=False)
            gust_ds = _as_dataset(raw)
            break
        except Exception:
            continue

    if gust_ds is None:
        raise RuntimeError(
            f"Could not find a gust/wind field in RAP wrfmsl "
            f"cycle={cycle_utc} fxx={fxx}"
        )

    # ── locate the gust / wind-speed variable ────────────────────────────────
    gust_var = None
    for vname, da in gust_ds.data_vars.items():
        short = (da.attrs.get("GRIB_shortName") or "").lower()
        name  = (da.attrs.get("GRIB_name") or vname).lower()
        if short in ("gust", "10u", "u10", "wind") or "gust" in name or "wind" in name:
            gust_var = vname
            break

    if gust_var is None:
        gust_var = list(gust_ds.data_vars)[0]   # last resort: first variable

    da_gust = gust_ds[gust_var]

    # ── latitude / longitude arrays ───────────────────────────────────────────
    lat2d = gust_ds["latitude"].values  if "latitude"  in gust_ds.coords else None
    lon2d = gust_ds["longitude"].values if "longitude" in gust_ds.coords else None

    if lat2d is None:
        raise RuntimeError("Cannot find latitude coordinate in RAP dataset.")

    # Normalise longitude to -180..180
    lon2d = np.where(lon2d > 180, lon2d - 360, lon2d)

    vals = da_gust.values.squeeze()

    # ── CONUS mask + stride ───────────────────────────────────────────────────
    mask = (
        (lat2d >= LAT_MIN) & (lat2d <= LAT_MAX) &
        (lon2d >= LON_MIN) & (lon2d <= LON_MAX)
    )

    # Apply stride to reduce point count
    # Build a strided mask over the 2-D grid
    strided = np.zeros_like(mask, dtype=bool)
    strided[::_STRIDE, ::_STRIDE] = True
    mask = mask & strided

    lats_flat  = lat2d[mask].ravel()
    lons_flat  = lon2d[mask].ravel()
    gusts_flat = vals[mask].ravel()

    # Convert m/s → knots; clamp negatives
    gusts_kt = np.maximum(gusts_flat * 1.94384, 0.0)

    def _cat(kt: float) -> int:
        if kt >= 50: return 3
        if kt >= 35: return 2
        if kt >= 20: return 1
        return 0

    points = [
        {
            "lat":     round(float(lats_flat[i]),  3),
            "lon":     round(float(lons_flat[i]),  3),
            "gust_kt": round(float(gusts_kt[i]),   1),
            "cat":     _cat(float(gusts_kt[i])),
        }
        for i in range(len(lats_flat))
        if np.isfinite(gusts_flat[i])
    ]

    cell_deg = 0.117 * _STRIDE   # ~13 km native * stride → degrees

    return {
        "model":       "RAP13",
        "product":     "GUST surface",
        "cycle_utc":   cycle_dt.replace(tzinfo=timezone.utc).isoformat(timespec="minutes"),
        "valid_utc":   valid_dt.isoformat(timespec="minutes").replace("+00:00", "Z"),
        "fxx":         fxx,
        "point_count": len(points),
        "cell_size_deg": round(cell_deg, 3),
        "points":      points,
    }


def get_rap_conus_cached(cycle_utc: str, fxx: int, ttl_seconds: int = 600) -> dict:
    """Return cached gust data or fetch fresh."""
    key = (cycle_utc, fxx)
    now = time.time()
    cached = _CACHE.get(key)
    if cached and (now - cached["ts"]) < ttl_seconds:
        return cached["data"]
    data = fetch_rap_conus_gusts(cycle_utc, fxx)
    _CACHE[key] = {"data": data, "ts": now}
    return data
