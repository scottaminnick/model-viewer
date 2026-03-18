"""
products/science/sigma_omega.py — Mountain wave σ(ω) computation helpers.

## Physical basis

Where mountain waves are active, the atmosphere has tightly alternating
regions of upward and downward motion (positive/negative omega) packed into
a small horizontal area.  The *spatial standard deviation* of omega in a
5×5 grid-point neighbourhood (~15 × 15 km at HRRR 3 km spacing) therefore
spikes wherever wave activity is present — even when the mean omega is near
zero — making it a clean, model-independent wave signal.

## Algorithm

For each target pressure level:

1. Fetch omega (VVEL, Pa s⁻¹) and geopotential height via Herbie.
2. Compute σ(ω) with scipy.ndimage.generic_filter (5×5 sliding window).
   This replaces the original nested Python loop and is ~100× faster.
3. Apply a light Gaussian smooth to reduce single-point speckle.

## Primary usage (CONUS overlays)

`_fetch_level()` and `_compute_stdev_omega()` are called by
`_SigmaOmegaLevel.get_values()` in products/definitions.py to produce
full-CONUS (lat2d, lon2d, stdev) arrays served via /api/image/ — one
product per pressure level, identical pipeline to Wind Gusts, Icing, etc.

Eight HRRR products are registered:
  hi set: 200, 250, 300, 400 hPa  (upper trop / tropopause)
  lo set: 500, 600, 700, 800 hPa  (mid/lower troposphere)

## Legacy composite helper

`fetch_sigma_omega_composite()` produces a 2×2 cartopy panel figure for
the Mountain West domain.  Retained for potential future use; the
/api/composite/ endpoint remains in app.py but is not linked from the UI.

Adapted from HRRR_four_level_stdev_omega_{lo,hi}_forecast scripts
(R. Connell) for the model-viewer Herbie pipeline.
"""

from __future__ import annotations

import io
import logging
from datetime import timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import generic_filter, gaussian_filter

from renderer import herbie_fetch, extract_var, get_latlon

log = logging.getLogger(__name__)

LEVEL_SETS: dict[str, list[int]] = {
    "lo": [500, 600, 700, 800],
    "hi": [200, 250, 300, 400],
}

_LAT_C = 40.4
_LON_C = -111.8
_DLAT  = 20.0
_LAT_S = _LAT_C - _DLAT / 2
_LAT_N = _LAT_C + _DLAT / 2
_DLON  = (5.0 / 4.0 * _DLAT) / np.cos(np.radians(_LAT_C))
_LON_W = _LON_C - _DLON / 2
_LON_E = _LON_C + _DLON / 2


def _fetch_level(
    herbie_model: str,
    herbie_product: str,
    cycle_dt,
    fxx: int,
    level_hpa: int,
    tag_base: str,
) -> tuple:
    """
    Fetch omega (VVEL, Pa s⁻¹) and geopotential height at one pressure level.

    Returns
    -------
    lat2d, lon2d : 2-D coordinate arrays
    omega        : vertical velocity (Pa s⁻¹)
    hght         : geopotential height (m)
    """
    ds_w = herbie_fetch(
        herbie_model, herbie_product, cycle_dt, fxx,
        [f":VVEL:{level_hpa} mb:"],
        f"{tag_base}_w{level_hpa}",
    )
    ds_z = herbie_fetch(
        herbie_model, herbie_product, cycle_dt, fxx,
        [f":HGT:{level_hpa} mb:"],
        f"{tag_base}_z{level_hpa}",
    )

    omega = extract_var(ds_w, ["w", "vvel", "VVEL", "wz", "dzdt"])
    hght  = extract_var(ds_z, ["gh", "z", "hgt", "HGT", "HGHT"])
    lat2d, lon2d = get_latlon(ds_w)

    return lat2d, lon2d, omega, hght


def _compute_stdev_omega(omega: np.ndarray, smooth_sigma: float = 2.0) -> np.ndarray:
    """
    Sliding 5×5 standard deviation of omega, then Gaussian smooth.

    The 5×5 window at HRRR 3 km spacing = ~15 km neighbourhood, matching
    the original script's 'radius=2' bounding box approach.
    generic_filter is equivalent to the original nested loop but ~100× faster.
    smooth_sigma=2 (≈ 6 km) removes sub-wave-scale speckle while preserving
    the wave envelope signature.
    """
    stdev = generic_filter(omega.astype(float), np.std, size=5, mode="nearest")
    return gaussian_filter(stdev, sigma=smooth_sigma)


def fetch_sigma_omega_composite(
    herbie_model: str,
    herbie_product: str,
    cycle_dt,
    fxx: int,
    level_set: str = "lo",
) -> bytes:
    """
    Compute σ(ω) at four pressure levels and return a styled 2×2 panel
    PNG as raw bytes, ready for direct HTTP response.

    Parameters
    ----------
    herbie_model   : "hrrr" (only model currently supported)
    herbie_product : "prs"
    cycle_dt       : model cycle datetime (naive UTC)
    fxx            : forecast hour
    level_set      : "lo" (500–800 hPa) or "hi" (200–400 hPa)
    """
    levels = LEVEL_SETS[level_set]
    tag    = (f"{herbie_model}_{cycle_dt.strftime('%Y%m%d%H')}"
              f"_{fxx:02d}_sigomega_{level_set}")

    if level_set == "lo":
        cf_levels = np.arange(0.20, 2.20, 0.20)
    else:
        cf_levels = np.arange(0.15, 1.65, 0.15)

    cmap = plt.get_cmap("hot_r")

    panels: dict[int, tuple] = {}
    for lvl in levels:
        log.info("sigma_omega: fetching %d hPa  F%02d  %s", lvl, fxx, level_set)
        lat2d, lon2d, omega, hght = _fetch_level(
            herbie_model, herbie_product, cycle_dt, fxx, lvl, tag
        )
        stdev_sm   = _compute_stdev_omega(omega)
        hght_100ft = gaussian_filter(hght * 3.28084 / 100.0, sigma=3.0)
        panels[lvl] = (lat2d, lon2d, stdev_sm, hght_100ft)

    fig, axarr = plt.subplots(
        nrows=2, ncols=2,
        figsize=(16, 14),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axlist = axarr.flatten()
    plt.subplots_adjust(wspace=0.06, hspace=0.22)
    fig.patch.set_facecolor("#0d1117")

    superneg1 = r"$^{-1}$"
    cf_ref = None

    for idx, lvl in enumerate(levels):
        ax = axlist[idx]
        lat2d, lon2d, stdev_sm, hght_100ft = panels[lvl]

        ax.set_extent([_LON_W, _LON_E, _LAT_S, _LAT_N], ccrs.PlateCarree())
        ax.set_facecolor("#0d1117")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        ax.add_feature(
            cfeature.STATES.with_scale("50m"),
            edgecolor="#888888", linewidth=0.9, zorder=5,
        )
        ax.add_feature(
            cfeature.LAKES.with_scale("50m"),
            facecolor="none", edgecolor="#888888", linewidth=0.6, zorder=5,
        )

        cF = ax.contourf(
            lon2d, lat2d, stdev_sm,
            levels=cf_levels, cmap=cmap, extend="max",
            transform=ccrs.PlateCarree(), zorder=3,
        )

        cint = 5.0 if lvl < 400 else 2.0
        cN = ax.contour(
            lon2d, lat2d, hght_100ft,
            levels=np.arange(0.0, 500.0, cint),
            colors="#000000", linewidths=1.6,
            transform=ccrs.PlateCarree(), zorder=7,
        )
        ax.clabel(cN, inline=True, fontsize=9, fmt="%i", zorder=8)

        ax.set_title(
            f"{lvl} hPa  Height (×100 ft)  &  σ(ω)  (Pa s{superneg1})",
            ha="left", loc="left", fontsize=11, color="#e6edf3",
        )

        if cf_ref is None:
            cf_ref = cF

    cb = fig.colorbar(
        cf_ref,
        orientation="horizontal",
        ax=axarr.ravel().tolist(),
        shrink=0.92, pad=0.04, aspect=65,
    )
    cb.set_ticks(cf_levels)
    cb.ax.tick_params(labelsize=10, colors="#e6edf3")
    cb.set_label(
        f"σ(ω)  in 5×5 Grid Box  (Pa s{superneg1})",
        fontsize=11, color="#e6edf3",
    )
    cb.outline.set_edgecolor("#30363d")

    valid_dt  = cycle_dt + timedelta(hours=fxx)
    valid_str = valid_dt.strftime("%H%M UTC  %a %b %d %Y")
    lvl_label = (
        "Lower Trop  (500–800 hPa)" if level_set == "lo"
        else "Upper Trop  (200–400 hPa)"
    )
    fig.suptitle(
        f"HRRR  σ(ω)  Mountain Wave  |  {lvl_label}"
        f"  |  F{fxx:02d}  |  Valid: {valid_str}",
        x=0.02, y=1.01, ha="left", fontsize=13, color="#e6edf3",
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                dpi=120, facecolor="#0d1117")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
