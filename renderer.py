"""
renderer.py — shared fetch / render / cache machinery for all model products.

Every product follows the same pipeline:
  1. Find latest cycle on AWS (Herbie inventory probe)
  2. Call product.get_values(cycle_dt, fxx) → (lat2d, lon2d, vals2d)
  3a. Render PNG via matplotlib pcolormesh  → served as imageOverlay
  3b. Extract point list                    → served as JSON for cursor sampling
  4. Cache results keyed by (model_id, product_id, cycle_utc, fxx)

Products only need to implement get_values() — everything else is here.
"""

import io
import os
import time
import warnings
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from herbie import Herbie

log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
HERBIE_DIR = Path(os.environ.get("HERBIE_DATA_DIR", "/tmp/herbie"))
HERBIE_DIR.mkdir(parents=True, exist_ok=True)

LAT_MIN, LAT_MAX =  22.0,  52.0
LON_MIN, LON_MAX = -126.0, -64.0


# ── Herbie helpers ────────────────────────────────────────────────────────────
def as_dataset(obj) -> xr.Dataset:
    """Normalize Herbie.xarray() return value to a single xr.Dataset."""
    if isinstance(obj, xr.Dataset):
        return obj
    if isinstance(obj, list):
        if not obj:
            raise ValueError("Herbie returned an empty list.")
        if all(isinstance(x, xr.Dataset) for x in obj):
            return xr.merge(obj, compat="override", combine_attrs="override")
        if all(isinstance(x, xr.DataArray) for x in obj):
            return xr.merge(
                [da.to_dataset(name=da.name or f"var_{i}") for i, da in enumerate(obj)],
                compat="override", combine_attrs="override",
            )
        return as_dataset(obj[0])
    raise TypeError(f"Unexpected Herbie return type: {type(obj)}")


def herbie_fetch(herbie_model: str, herbie_product: str,
                 cycle_dt: datetime, fxx: int,
                 searches: list[str], subdir: str) -> xr.Dataset:
    """
    Try each search string in order until one returns data.
    Uses an isolated save_dir per (model, product, cycle, fxx, field)
    to prevent cfgrib index hash collisions between concurrent products.
    """
    save_dir = HERBIE_DIR / subdir
    save_dir.mkdir(parents=True, exist_ok=True)

    H = Herbie(cycle_dt, model=herbie_model, product=herbie_product,
               fxx=fxx, save_dir=str(save_dir), overwrite=False)

    for search in searches:
        try:
            raw = H.xarray(search, remove_grib=False)
            return as_dataset(raw)
        except Exception as e:
            log.debug(f"Search '{search}' failed: {e}")

    raise RuntimeError(
        f"No GRIB field found for {herbie_model}/{herbie_product} "
        f"fxx={fxx} with searches={searches}"
    )


def extract_var(ds: xr.Dataset, hints: list[str]) -> np.ndarray:
    """
    Find the first data variable whose shortName or name matches any hint.
    Falls back to the first variable if nothing matches.
    """
    for vname, da in ds.data_vars.items():
        short = (da.attrs.get("GRIB_shortName") or "").lower()
        name  = (da.attrs.get("GRIB_name")      or "").lower()
        if any(h in short or h in name for h in hints):
            return da.values.squeeze()
    return list(ds.data_vars.values())[0].values.squeeze()


def get_latlon(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Return (lat2d, lon2d) from dataset, normalising lon to -180..180."""
    lat2d = ds["latitude"].values
    lon2d = ds["longitude"].values
    lon2d = np.where(lon2d > 180, lon2d - 360, lon2d)
    return lat2d, lon2d


# ── cycle / status helpers ────────────────────────────────────────────────────
def find_latest_cycle(herbie_model: str, herbie_product: str,
                      max_lookback: int = 6) -> datetime:
    """Walk backwards from the current hour until we find a live cycle."""
    base = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    probe_dir = HERBIE_DIR / f"{herbie_model}_probe"
    for h in range(max_lookback + 1):
        candidate = base - timedelta(hours=h)
        try:
            H = Herbie(candidate, model=herbie_model, product=herbie_product,
                       fxx=1, save_dir=str(probe_dir), overwrite=False)
            H.inventory()
            return candidate
        except Exception:
            continue
    return base - timedelta(hours=1)


def get_cycle_status(model_id: str, herbie_model: str, herbie_product: str,
                     fxx_max: int, cached_keys: set,
                     _status_cache: dict, ttl: int = 300) -> dict:
    """
    Return a status dict listing available forecast hours for the
    three most recent cycles. Cached for `ttl` seconds.
    """
    now = time.time()
    if _status_cache.get("data") and (now - _status_cache.get("ts", 0)) < ttl:
        return _status_cache["data"]

    latest = find_latest_cycle(herbie_model, herbie_product)
    probe_dir = HERBIE_DIR / f"{herbie_model}_probe"
    cycles = []

    for offset in range(3):
        cycle_dt  = latest - timedelta(hours=offset)
        cycle_str = cycle_dt.replace(tzinfo=timezone.utc).isoformat(timespec="minutes")
        available = []
        for fxx in range(0, fxx_max + 1):
            try:
                H = Herbie(cycle_dt, model=herbie_model, product=herbie_product,
                           fxx=fxx, save_dir=str(probe_dir), overwrite=False)
                H.inventory()
                available.append(fxx)
            except Exception:
                if fxx > 2:   # stop after first gap beyond F02
                    break
        pct = round(len(available) / fxx_max * 100)
        cycles.append({
            "cycle_utc":       cycle_str,
            "available_hours": available,
            "pct_complete":    pct,
            "cached_hours":    [fxx for fxx in available
                                 if (cycle_str, fxx) in cached_keys],
        })

    result = {"model": model_id, "cycles": cycles}
    _status_cache["data"] = result
    _status_cache["ts"]   = now
    return result


# ── PNG renderer ──────────────────────────────────────────────────────────────
def render_png(lat2d: np.ndarray, lon2d: np.ndarray, vals2d: np.ndarray,
               cmap: mcolors.ListedColormap,
               norm: mcolors.BoundaryNorm,
               render_mode: str = "fill") -> bytes:
    """
    Render a 2-D field to a transparent PNG that exactly covers
    [LAT_MIN,LAT_MAX] x [LON_MIN,LON_MAX] — ready for L.imageOverlay.
    """
    # Clip to CONUS domain
    col_mask = (lon2d[0, :] >= LON_MIN - 1) & (lon2d[0, :] <= LON_MAX + 1)
    row_mask = (lat2d[:, 0] >= LAT_MIN - 1) & (lat2d[:, 0] <= LAT_MAX + 1)
    lon2d  = lon2d[np.ix_(row_mask, col_mask)]
    lat2d  = lat2d[np.ix_(row_mask, col_mask)]
    vals2d = vals2d[np.ix_(row_mask, col_mask)]

    fig = plt.figure(figsize=(18, 10), dpi=120)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect("auto")
    ax.axis("off")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if render_mode == "contour":
            levels = norm.boundaries
            ax.contour(lon2d, lat2d, vals2d,
                       levels=levels, colors=["#ffffff"],
                       linewidths=1.2, alpha=0.85)
            ax.contourf(lon2d, lat2d, vals2d,
                        levels=levels, cmap=cmap, norm=norm, alpha=0.65)
        else:  # "fill"
            ax.pcolormesh(lon2d, lat2d, vals2d,
                          cmap=cmap, norm=norm,
                          shading="nearest", rasterized=True)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True,
                bbox_inches=None, pad_inches=0, dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── point extractor ───────────────────────────────────────────────────────────
def extract_points(lat2d: np.ndarray, lon2d: np.ndarray, vals2d: np.ndarray,
                   stride: int = 2) -> list[dict]:
    """
    Subsample the grid and return a list of {lat, lon, value} dicts
    covering the CONUS domain.  Used for cursor-sampling in the browser.
    """
    mask = (
        (lat2d >= LAT_MIN) & (lat2d <= LAT_MAX) &
        (lon2d >= LON_MIN) & (lon2d <= LON_MAX)
    )
    strided = np.zeros_like(mask, dtype=bool)
    strided[::stride, ::stride] = True
    mask = mask & strided

    lats = lat2d[mask].ravel()
    lons = lon2d[mask].ravel()
    vals = vals2d[mask].ravel()

    return [
        {"lat": round(float(lats[i]), 3),
         "lon": round(float(lons[i]), 3),
         "value": round(float(vals[i]), 1)}
        for i in range(len(lats))
        if np.isfinite(vals[i])
    ]


# ── generic TTL cache ─────────────────────────────────────────────────────────
class TTLCache:
    """
    Thread-safe in-memory cache keyed by (model_id, product_id, cycle_utc, fxx).
    Stores arbitrary payloads with a timestamp for TTL expiry.
    """
    def __init__(self):
        self._store: dict = {}

    def _key(self, model_id, product_id, cycle_utc, fxx):
        return (model_id, product_id, cycle_utc, int(fxx))

    def get(self, model_id, product_id, cycle_utc, fxx, ttl):
        k = self._key(model_id, product_id, cycle_utc, fxx)
        entry = self._store.get(k)
        if entry and (time.time() - entry["ts"]) < ttl:
            return entry["data"]
        return None

    def set(self, model_id, product_id, cycle_utc, fxx, data):
        k = self._key(model_id, product_id, cycle_utc, fxx)
        self._store[k] = {"data": data, "ts": time.time()}

    def keys(self) -> set:
        """Return all (cycle_utc, fxx) pairs currently cached."""
        return {(k[2], k[3]) for k in self._store}


# Singletons shared across all products
IMAGE_CACHE  = TTLCache()
POINTS_CACHE = TTLCache()
