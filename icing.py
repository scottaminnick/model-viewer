"""
icing.py  –  HRRR Winter Icing Threat Index for Colorado
==========================================================
Adapted from WinterIcingThreat.py (GFE Smart Tool).

Ingredients
-----------
  Saturation   min(RH850, RH700) – deep-layer moisture saturation proxy
  Ascent       min(VVEL850, VVEL700) – omega in Pa/s (negative = ascent)
  Convergence  -div(U850,V850) – low-level convergence as frontogenesis proxy
  Upslope      850 mb wind direction + speed modifier for Colorado terrain:
                 Front Range:  E/SE sector  045–135°, ≥10 kt  → +0.15
                 West slope:   W/NW sector  225–315°, ≥10 kt  → +0.10

Scoring weights (tunable at bottom of file):
  score = 0.45·sat + 0.35·ascent + 0.20·conv + upslope

Categories
----------
  0  Green   score < 0.35
  1  Yellow  0.35 ≤ score < 0.55
  2  Orange  0.55 ≤ score < 0.75
  3  Red     score ≥ 0.75

GRIB download
-------------
  Product: wrfprsf##.grib2 (prs)
  Fields:  RH, UGRD, VGRD, VVEL at 850 mb and 700 mb  = 8 messages
  Typical download size: ~3–5 MB (byte-range subset via IDX)
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
_MAX_SUBSET_MB = 50   # if larger, NOMADS returned full file

# ── GRIB search string ────────────────────────────────────────────────────────
# Non-capturing groups → matches exactly 8 messages:
#   RH:850 mb, RH:700 mb
#   UGRD:850 mb, UGRD:700 mb
#   VGRD:850 mb, VGRD:700 mb
#   VVEL:850 mb, VVEL:700 mb
PRS_SEARCH = r"(?:RH|UGRD|VGRD|VVEL):(?:850|700) mb"

# ── In-memory cache keyed by (cycle_utc, fxx) ────────────────────────────────
_CACHE    = {}
_CLIP_IDX = {}   # cached clip indices by grid shape


# ── Scoring weights (tune here without touching the algorithm) ────────────────
W_SAT     = 0.45   # saturation
W_ASCENT  = 0.35   # vertical motion
W_CONV    = 0.20   # low-level convergence (frontogenesis proxy)

# Upslope modifiers
UPSLOPE_FRONT_RANGE = 0.15   # E/SE 045–135°
UPSLOPE_WEST_SLOPE  = 0.10   # W/NW 225–315°
UPSLOPE_SPD_KT      = 10.0   # minimum 850 mb wind speed (kt) for modifier

# Category thresholds (match GFE tool)
CAT_YELLOW = 0.35
CAT_ORANGE = 0.55
CAT_RED    = 0.75


# ── Herbie helpers ────────────────────────────────────────────────────────────

def _now_utc_hour_naive():
    return datetime.utcnow().replace(minute=0, second=0, microsecond=0)


def _find_latest_hrrr_cycle(max_lookback: int = 6) -> datetime:
    base = _now_utc_hour_naive()
    for h in range(max_lookback + 1):
        dt = base - timedelta(hours=h)
        try:
            H = Herbie(dt, model="hrrr", product="prs", fxx=1,
                       save_dir=str(HERBIE_DIR), overwrite=False)
            H.inventory()
            return dt
        except Exception:
            continue
    return base


def _download_subset(cycle: datetime, fxx: int) -> Path:
    H = Herbie(cycle, model="hrrr", product="prs", fxx=fxx,
               save_dir=str(HERBIE_DIR), overwrite=False)
    result = H.download(searchString=PRS_SEARCH)
    p = Path(result) if result else None
    if p is None or not p.exists():
        raise FileNotFoundError(f"Download failed for prs {cycle} F{fxx:02d}")
    size_mb = p.stat().st_size / 1_000_000
    if size_mb > _MAX_SUBSET_MB:
        raise RuntimeError(
            f"Downloaded file is {size_mb:.0f} MB — NOMADS returned full file. "
            f"Retry when data migrates to AWS (~1–2 hrs after cycle)."
        )
    return p


# ── Grid clipping helpers ─────────────────────────────────────────────────────

def _get_clip_idx(lat2d, lon2d):
    shape_key = lat2d.shape
    if shape_key in _CLIP_IDX:
        return _CLIP_IDX[shape_key]
    mask_lat = (lat2d >= CO_LAT_MIN) & (lat2d <= CO_LAT_MAX)
    mask_lon = (lon2d >= CO_LON_MIN) & (lon2d <= CO_LON_MAX)
    mask = mask_lat & mask_lon
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    step = 2   # every-other point ≈ 6 km spacing for HRRR 3 km grid
    idx = (r0, r1, c0, c1, step)
    _CLIP_IDX[shape_key] = idx
    return idx


def _clip(data, idx):
    r0, r1, c0, c1, step = idx
    return data[r0:r1, c0:c1][::step, ::step].astype(np.float32)


# ── GRIB reader ───────────────────────────────────────────────────────────────

def _read_prs_fields(subset_path: Path):
    """
    Single pass through the small prs subset.
    Returns clipped Colorado arrays for:
        RH850, RH700, U850, V850, U700, V700, VVEL850, VVEL700
    plus lat_co, lon_co.

    GRIB name mapping used by HRRR:
        'Relative humidity'          → RH
        'U component of wind'        → UGRD
        'V component of wind'        → VGRD
        'Vertical velocity'          → VVEL  (Pa/s, negative = ascent)
    """
    want = {
        ("Relative humidity",    850): "RH850",
        ("Relative humidity",    700): "RH700",
        ("U component of wind",  850): "U850",
        ("V component of wind",  850): "V850",
        ("U component of wind",  700): "U700",
        ("V component of wind",  700): "V700",
        ("Vertical velocity",    850): "VVEL850",
        ("Vertical velocity",    700): "VVEL700",
    }
    fields   = {}
    lat_co   = lon_co = None
    clip_idx = None

    grbs = pygrib.open(str(subset_path))
    for grb in grbs:
        if grb.typeOfLevel != "isobaricInhPa":
            continue
        key = (grb.name, grb.level)
        if key not in want:
            continue
        data, lat2d, lon2d = grb.data()
        lon2d = np.where(lon2d > 180, lon2d - 360, lon2d)

        if clip_idx is None:
            clip_idx = _get_clip_idx(lat2d, lon2d)
            r0, r1, c0, c1, step = clip_idx
            lat_co = lat2d[r0:r1, c0:c1][::step, ::step]
            lon_co = lon2d[r0:r1, c0:c1][::step, ::step]

        fields[want[key]] = _clip(data, clip_idx)
        del data, lat2d, lon2d

    grbs.close()
    gc.collect()

    missing = [k for k in want.values() if k not in fields]
    if missing:
        raise ValueError(f"Missing prs fields: {missing} — check GRIB search string")

    return (lat_co, lon_co,
            fields["RH850"],   fields["RH700"],
            fields["U850"],    fields["V850"],
            fields["U700"],    fields["V700"],
            fields["VVEL850"], fields["VVEL700"])


# ── Science functions ─────────────────────────────────────────────────────────

def _saturation_score(rh850, rh700):
    """
    Deep-layer moisture saturation 0→1.
    Mirrors GFE: score ramps from 80% → 100% RH.
    Uses the MINIMUM of the two levels (both must be saturated).
    """
    rh_min = np.minimum(rh850, rh700)
    return np.clip((rh_min - 80.0) / 20.0, 0.0, 1.0)


def _ascent_score(vvel850, vvel700):
    """
    Vertical motion score 0→1.
    VVEL in Pa/s — negative values = upward motion (ascent).
    Thresholds: -0.1 Pa/s (weak) → -0.5 Pa/s (strong).
    Mirrors GFE omega logic converted from ubar/s to Pa/s.
    """
    om_min = np.minimum(vvel850, vvel700)   # most negative = strongest ascent
    return np.clip((-om_min - 0.1) / 0.4, 0.0, 1.0)


def _convergence_score(u850, v850):
    """
    Low-level convergence as frontogenesis proxy.
    Convergence = -divergence = -(∂U/∂x + ∂V/∂y).
    We use numpy.gradient for finite differences over the grid.
    Thresholds: 0.5e-5 s⁻¹ (weak) → 2.0e-5 s⁻¹ (strong convergence).
    """
    # Approximate grid spacing at ~39°N for HRRR 3km (subsampled 2x → 6km)
    dx = 6000.0   # metres
    dy = 6000.0
    du_dy, du_dx = np.gradient(u850, dy, dx)
    dv_dy, dv_dx = np.gradient(v850, dy, dx)
    divergence  = du_dx + dv_dy
    convergence = -divergence                 # positive = convergent
    return np.clip((convergence - 0.5e-5) / 1.5e-5, 0.0, 1.0)


def _upslope_modifier(u850, v850):
    """
    Colorado terrain-specific upslope wind modifier.

    Front Range  – easterly upslope:  850 mb wind from 045°–135°, ≥10 kt
                   → add UPSLOPE_FRONT_RANGE (0.15)
    West slope   – westerly upslope:  850 mb wind from 225°–315°, ≥10 kt
                   → add UPSLOPE_WEST_SLOPE  (0.10)

    Both can activate simultaneously in different parts of the grid
    (e.g. a strong upper-level trough driving E flow on the plains
    while westerlies persist over the western slope).
    """
    # Wind speed in kt (U/V from HRRR are m/s → ×1.94384)
    spd_kt = np.sqrt(u850**2 + v850**2) * 1.94384

    # Meteorological wind direction: 0° = from N, 90° = from E
    # arctan2(U, V) gives direction wind is FROM
    wdir = (np.degrees(np.arctan2(u850, v850)) + 360.0) % 360.0

    front_range = (
        (wdir >= 45.0) & (wdir <= 135.0) &
        (spd_kt >= UPSLOPE_SPD_KT)
    )
    west_slope = (
        (wdir >= 225.0) & (wdir <= 315.0) &
        (spd_kt >= UPSLOPE_SPD_KT)
    )

    modifier = np.zeros_like(spd_kt, dtype=np.float32)
    modifier[front_range] += UPSLOPE_FRONT_RANGE
    modifier[west_slope]  += UPSLOPE_WEST_SLOPE
    return modifier


def _composite_score(sat, ascent, conv, modifier):
    return W_SAT * sat + W_ASCENT * ascent + W_CONV * conv + modifier


def _categorise(score):
    cat = np.zeros_like(score, dtype=np.int8)
    cat[score >= CAT_YELLOW] = 1
    cat[score >= CAT_ORANGE] = 2
    cat[score >= CAT_RED]    = 3
    return cat


# ── Main fetch function ───────────────────────────────────────────────────────

def fetch_icing(cycle_utc: str, fxx: int = 1) -> dict:
    """
    Compute Winter Icing Threat Index over Colorado.

    cycle_utc : ISO string e.g. '2026-02-24T02:00Z'
    fxx       : HRRR forecast hour (1–12)

    Returns a dict ready to jsonify with keys:
        points, point_count, valid_utc, cycle_utc, fxx,
        weights, thresholds
    """
    cycle = datetime.fromisoformat(
        cycle_utc.replace("Z", "+00:00")
    ).replace(tzinfo=None)
    cycle_aware = cycle.replace(tzinfo=timezone.utc)

    # ── Download + read under global lock ────────────────────────────────────
    if not GRIB_LOCK.acquire(timeout=30):
        raise RuntimeError(
            "GRIB_LOCK timeout — another download is in progress, retry in a moment."
        )
    try:
        prs_path = _download_subset(cycle, fxx)
        (lat_co, lon_co,
         rh850, rh700,
         u850, v850,
         u700, v700,
         vvel850, vvel700) = _read_prs_fields(prs_path)
    finally:
        GRIB_LOCK.release()

    # ── Score ingredients ─────────────────────────────────────────────────────
    sat      = _saturation_score(rh850, rh700)
    ascent   = _ascent_score(vvel850, vvel700)
    conv     = _convergence_score(u850, v850)
    modifier = _upslope_modifier(u850, v850)
    score    = _composite_score(sat, ascent, conv, modifier)
    cat      = _categorise(score)

    # 850 mb wind speed (kt) for popup display
    spd850_kt = np.sqrt(u850**2 + v850**2) * 1.94384
    wdir850   = (np.degrees(np.arctan2(u850, v850)) + 360.0) % 360.0

    # ── Build output points ───────────────────────────────────────────────────
    ny, nx = lat_co.shape
    points = []
    for i in range(ny):
        for j in range(nx):
            s = float(score[i, j])
            points.append({
                "lat":      round(float(lat_co[i, j]), 4),
                "lon":      round(float(lon_co[i, j]), 4),
                "score":    round(s, 3),
                "cat":      int(cat[i, j]),
                "rh850":    round(float(rh850[i, j]), 1),
                "rh700":    round(float(rh700[i, j]), 1),
                "sat":      round(float(sat[i, j]), 3),
                "ascent":   round(float(ascent[i, j]), 3),
                "conv":     round(float(conv[i, j]), 3),
                "spd850":   round(float(spd850_kt[i, j]), 1),
                "wdir850":  round(float(wdir850[i, j]), 0),
            })

    valid_dt  = cycle + timedelta(hours=fxx)
    valid_utc = (valid_dt.replace(tzinfo=timezone.utc)
                 .isoformat(timespec="minutes")
                 .replace("+00:00", "Z"))

    return {
        "points":      points,
        "point_count": len(points),
        "valid_utc":   valid_utc,
        "cycle_utc":   cycle_aware.isoformat(timespec="minutes").replace("+00:00", "Z"),
        "fxx":         fxx,
        "cell_size_deg": 0.054,   # approximate cell size for renderer
        # Metadata useful for UI / tuning
        "weights":     {"sat": W_SAT, "ascent": W_ASCENT, "conv": W_CONV},
        "thresholds":  {"yellow": CAT_YELLOW, "orange": CAT_ORANGE, "red": CAT_RED},
    }


def get_icing_cached(cycle_utc: str, fxx: int = 1, ttl_seconds: int = 600) -> dict:
    """Cache keyed by (cycle_utc, fxx). Re-fetches after ttl_seconds."""
    key    = (cycle_utc, fxx)
    now    = time.time()
    cached = _CACHE.get(key)
    if cached is None or (now - cached["ts"]) > ttl_seconds:
        _CACHE[key] = {"ts": now, "data": fetch_icing(cycle_utc=cycle_utc, fxx=fxx)}
    return _CACHE[key]["data"]
