"""
products/definitions.py — all model × product definitions.

Adding a new product:
  1. Define colourmap + legend
  2. Instantiate ProductDef (or subclass)
  3. Call register(product)

Adding a new model (GFS, RRFS):
  - Change herbie_model / herbie_product on a copy of an existing def
  - Register under a new model_id
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import matplotlib.colors as mcolors

from products import ProductDef, register
from renderer import herbie_fetch, extract_var, get_latlon
from icing_threat import PRS_SEARCH

# ══════════════════════════════════════════════════════════════════════════════
#  Helper: build a ListedColormap + BoundaryNorm + legend from parallel lists
# ══════════════════════════════════════════════════════════════════════════════

def _scale(bounds: list, colors: list, labels: list) -> tuple:
    """Return (cmap, norm, legend_list)."""
    cmap   = mcolors.ListedColormap(colors)
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)
    legend = [{"color": c, "label": l} for c, l in zip(colors, labels)]
    return cmap, norm, legend


# ══════════════════════════════════════════════════════════════════════════════
#  WIND GUSTS — surface   (RAP13 + HRRR)
# ══════════════════════════════════════════════════════════════════════════════

_gust_cmap, _gust_norm, _gust_legend = _scale(
    bounds  = [0, 5, 10, 15, 20, 25, 35, 50, 200],
    colors  = ['#4575b4','#74add1','#abd9e9','#e0f3f8',
               '#fee090','#fc8d59','#d73027','#a50026'],
    labels  = ['<5 kt — Calm','5–10 kt — Light','10–15 kt — Breezy',
               '15–20 kt — Moderate','20–25 kt — Fresh','25–35 kt — Strong',
               '35–50 kt — Very Strong','≥50 kt — Extreme'],
)

@dataclass
class _WindGust(ProductDef):
    _var_hints: list = field(default_factory=lambda: ["gust","wind"])
    _units_fn:  object = field(default=lambda v: np.maximum(v * 1.94384, 0.0))

register(_WindGust(
    model_id="rap13", product_id="wind_gust",
    label="Wind Gusts — Surface", units="kt",
    herbie_model="rap", herbie_product="awp130pgrb",
    searches=[":GUST:surface:"],
    cmap=_gust_cmap, norm=_gust_norm, legend=_gust_legend,
))
register(_WindGust(
    model_id="hrrr", product_id="wind_gust",
    label="Wind Gusts — Surface", units="kt",
    herbie_model="hrrr", herbie_product="sfc",
    searches=[":GUST:surface:"],
    cmap=_gust_cmap, norm=_gust_norm, legend=_gust_legend,
))

# ══════════════════════════════════════════════════════════════════════════════
#  SURFACE WIND SPEED — 10m   (RAP13 + HRRR)
# ══════════════════════════════════════════════════════════════════════════════

_wind10_cmap, _wind10_norm, _wind10_legend = _scale(
    bounds  = [0, 5, 10, 15, 20, 25, 35, 50, 200],
    colors  = ['#f7f7f7','#d9f0a3','#addd8e','#78c679',
               '#41ab5d','#238443','#006837','#004529'],
    labels  = ['<5 kt','5–10 kt','10–15 kt','15–20 kt',
               '20–25 kt','25–35 kt','35–50 kt','≥50 kt'],
)

@dataclass
class _SurfaceWindSpeed(ProductDef):
    """Fetches 10m U+V in ONE call (paired GRIB message), returns wind speed in knots."""
    def get_values(self, cycle_dt, fxx):
        tag = f"{self.model_id}_{cycle_dt.strftime('%Y%m%d%H')}_{fxx:02d}_sfcwind"
        ds = herbie_fetch(self.herbie_model, self.herbie_product,
                          cycle_dt, fxx,
                          [":UGRD:10 m above ground:|:VGRD:10 m above ground:"],
                          tag + "_uv")
        u = extract_var(ds, ["ugrd", "u10", "u"])
        v = extract_var(ds, ["vgrd", "v10", "v"])
        lat2d, lon2d = get_latlon(ds)
        return lat2d, lon2d, np.sqrt(u**2 + v**2) * 1.94384

    def get_barb_data(self, cycle_dt, fxx):
        tag = f"{self.model_id}_{cycle_dt.strftime('%Y%m%d%H')}_{fxx:02d}_sfcwind"
        ds = herbie_fetch(self.herbie_model, self.herbie_product,
                          cycle_dt, fxx,
                          [":UGRD:10 m above ground:|:VGRD:10 m above ground:"],
                          tag + "_uv")
        u = extract_var(ds, ["ugrd", "u10", "u"])
        v = extract_var(ds, ["vgrd", "v10", "v"])
        lat2d, lon2d = get_latlon(ds)
        return lat2d, lon2d, u, v

register(_SurfaceWindSpeed(
    model_id="rap13", product_id="surface_wind",
    label="Surface Wind Speed — 10m", units="kt",
    herbie_model="rap", herbie_product="awp130pgrb",
    searches=[],  # get_values() handles fetching directly
    cmap=_wind10_cmap, norm=_wind10_norm, legend=_wind10_legend,
    supports_barbs=True,
))
register(_SurfaceWindSpeed(
    model_id="hrrr", product_id="surface_wind",
    label="Surface Wind Speed — 10m", units="kt",
    herbie_model="hrrr", herbie_product="sfc",
    searches=[],
    cmap=_wind10_cmap, norm=_wind10_norm, legend=_wind10_legend,
    supports_barbs=True,
    barb_stride=24,
))


# ══════════════════════════════════════════════════════════════════════════════
#  MSLP   (RAP13 + HRRR)   — contour rendering
# ══════════════════════════════════════════════════════════════════════════════

_mslp_bounds = list(range(940, 1060, 4))    # every 4 hPa
_n = len(_mslp_bounds) - 1
import matplotlib.cm as _cm
_mslp_cmap  = mcolors.ListedColormap([_cm.RdBu_r(i / _n) for i in range(_n)])
_mslp_norm  = mcolors.BoundaryNorm(_mslp_bounds, _mslp_cmap.N)
_mslp_legend = [{"color": "#4575b4", "label": "Low pressure"},
                {"color": "#d73027", "label": "High pressure"}]

@dataclass
class _MSLP(ProductDef):
    _var_hints: list  = field(default_factory=lambda: ["mslma","prmsl","msl"])
    _units_fn:  object = field(default=lambda v: v / 100.0)   # Pa → hPa

register(_MSLP(
    model_id="rap13", product_id="mslp",
    label="MSLP", units="hPa", render_mode="contour",
    herbie_model="rap", herbie_product="awp130pgrb",
    searches=[":MSLMA:mean sea level:", ":PRMSL:mean sea level:"],
    cmap=_mslp_cmap, norm=_mslp_norm, legend=_mslp_legend,
))
register(_MSLP(
    model_id="hrrr", product_id="mslp",
    label="MSLP", units="hPa", render_mode="contour",
    herbie_model="hrrr", herbie_product="sfc",
    searches=[":MSLMA:mean sea level:", ":PRMSL:mean sea level:"],
    cmap=_mslp_cmap, norm=_mslp_norm, legend=_mslp_legend,
))


# ══════════════════════════════════════════════════════════════════════════════
#  500mb WIND SPEED   (RAP13 + HRRR)
# ══════════════════════════════════════════════════════════════════════════════

_w500_cmap, _w500_norm, _w500_legend = _scale(
    bounds  = [0, 20, 40, 60, 80, 100, 120, 150, 300],
    colors  = ['#f7fbff','#c6dbef','#9ecae1','#6baed6',
               '#4292c6','#2171b5','#08519c','#08306b'],
    labels  = ['<20 kt','20–40 kt','40–60 kt','60–80 kt',
               '80–100 kt','100–120 kt','120–150 kt','≥150 kt'],
)

@dataclass
class _Wind500mb(ProductDef):
    """500mb wind speed — U+V fetched together."""
    def get_values(self, cycle_dt, fxx):
        tag = f"{self.model_id}_{cycle_dt.strftime('%Y%m%d%H')}_{fxx:02d}_500mb"
        ds = herbie_fetch(self.herbie_model, self.herbie_product,
                          cycle_dt, fxx,
                          [":UGRD:500 mb:|:VGRD:500 mb:"],
                          tag + "_uv")
        u = extract_var(ds, ["ugrd", "u"])
        v = extract_var(ds, ["vgrd", "v"])
        lat2d, lon2d = get_latlon(ds)
        return lat2d, lon2d, np.sqrt(u**2 + v**2) * 1.94384

    def get_barb_data(self, cycle_dt, fxx):
        tag = f"{self.model_id}_{cycle_dt.strftime('%Y%m%d%H')}_{fxx:02d}_500mb"
        ds = herbie_fetch(self.herbie_model, self.herbie_product,
                          cycle_dt, fxx,
                          [":UGRD:500 mb:|:VGRD:500 mb:"],
                          tag + "_uv")
        u = extract_var(ds, ["ugrd", "u"])
        v = extract_var(ds, ["vgrd", "v"])
        lat2d, lon2d = get_latlon(ds)
        return lat2d, lon2d, u, v

register(_Wind500mb(
    model_id="rap13", product_id="wind_500mb",
    label="500mb Wind Speed", units="kt",
    herbie_model="rap", herbie_product="awp130pgrb",
    searches=[],
    cmap=_w500_cmap, norm=_w500_norm, legend=_w500_legend,
    supports_barbs=True,
))
register(_Wind500mb(
    model_id="hrrr", product_id="wind_500mb",
    label="500mb Wind Speed", units="kt",
    herbie_model="hrrr", herbie_product="prs",
    searches=[],
    cmap=_w500_cmap, norm=_w500_norm, legend=_w500_legend,
    supports_barbs=True,
    barb_stride=24,
))


# ══════════════════════════════════════════════════════════════════════════════
#  MIXING HEIGHT (HPBL)   (RAP13 + HRRR)
# ══════════════════════════════════════════════════════════════════════════════

_mix_cmap, _mix_norm, _mix_legend = _scale(
    bounds  = [0, 650, 1650, 3300, 5000, 6600, 10000, 13000, 26000],
    colors  = ['#ffffcc','#d9f0a3','#addd8e','#78c679',
               '#41ab5d','#238443','#006837','#004529'],
    labels  = ['<650 ft','650–1650 ft','1650–3300 ft','3300–5000 ft',
               '5000–6600 ft','6600–10000 ft','10000–13000 ft','≥13000 ft'],
)
@dataclass
class _MixHeight(ProductDef):
    _var_hints: list  = field(default_factory=lambda: ["hpbl","pblh","mix"])
    _units_fn:  object = field(default=lambda v: v)

    def get_values(self, cycle_dt, fxx):
        from renderer import herbie_fetch, extract_var, get_latlon
        tag = f"{self.model_id}_{cycle_dt.strftime('%Y%m%d%H')}_{fxx:02d}_mixhgt"
        ds = herbie_fetch(self.herbie_model, self.herbie_product,
                          cycle_dt, fxx, self.searches, tag)
        vals = extract_var(ds, ["hpbl","pblh","mix"])
        lat2d, lon2d = get_latlon(ds)
        return lat2d, lon2d, np.maximum(vals, 0.0) * 3.28084

    def get_point_values(self, cycle_dt, fxx):
        from renderer import herbie_fetch, extract_var, get_latlon
        tag = f"{self.model_id}_{cycle_dt.strftime('%Y%m%d%H')}_{fxx:02d}_mixhgt"
        # Fetch HPBL
        ds = herbie_fetch(self.herbie_model, self.herbie_product,
                          cycle_dt, fxx, self.searches, tag)
        vals = extract_var(ds, ["hpbl","pblh","mix"])
        agl_ft = np.maximum(vals, 0.0) * 3.28084
        # Fetch orography from same product
        ds_orog = herbie_fetch(self.herbie_model, self.herbie_product,
                               cycle_dt, fxx, [":HGT:surface:"], tag + "_orog")
        orog = extract_var(ds_orog, ["orog","hgt","z"])
        terrain_ft = orog * 3.28084
        lat2d, lon2d = get_latlon(ds)
        return {"value": agl_ft, "msl_ft": agl_ft + terrain_ft}


register(_MixHeight(
    model_id="rap13", product_id="mix_height",
    label="Mixing Height (HPBL)", units="ft",
    herbie_model="rap", herbie_product="awp130pgrb",
    searches=[":HPBL:surface:"],
    cmap=_mix_cmap, norm=_mix_norm, legend=_mix_legend,
))
register(_MixHeight(
    model_id="hrrr", product_id="mix_height",
    label="Mixing Height (HPBL)", units="ft",
    herbie_model="hrrr", herbie_product="sfc",
    searches=[":HPBL:surface:"],
    cmap=_mix_cmap, norm=_mix_norm, legend=_mix_legend,
))

# ══════════════════════════════════════════════════════════════════════════════
#  ICING THREAT (RAP13 + HRRR)
# ══════════════════════════════════════════════════════════════════════════════
_ice_cmap, _ice_norm, _ice_legend = _scale(
    bounds = [0.0, 0.35, 0.55, 0.75, 1.2],
    colors = ['#f7fbff','#bdd7e7','#6baed6','#08306b'],
    labels = ['None','Light (≥0.35)','Moderate (≥0.55)','Heavy (≥0.75)'],
)
@dataclass
class _Icing(ProductDef):        # ← this line is missing
    def get_values(self, cycle_dt, fxx):
        from icing_threat import fetch_icing_arrays
        sfc = "sfc" if self.herbie_model == "hrrr" else "wrfsml"
        lat2d, lon2d, score2d = fetch_icing_arrays(
            self.herbie_model, self.herbie_product, cycle_dt, fxx,
            sfc_product="wrfmsl"
        )
        return lat2d, lon2d, score2d

register(_Icing(
    model_id="rap13", product_id="icing",
    label="Icing Threat (Winter)", units="index",
    herbie_model="rap", herbie_product="awp130pgrb",
    searches=[PRS_SEARCH],
    cmap=_ice_cmap, norm=_ice_norm, legend=_ice_legend,
))
register(_Icing(
    model_id="hrrr", product_id="icing",
    label="Icing Threat (Winter)", units="index",
    herbie_model="hrrr", herbie_product="prs",
    searches=[PRS_SEARCH],
    cmap=_ice_cmap, norm=_ice_norm, legend=_ice_legend,
))

# ══════════════════════════════════════════════════════════════════════════════
#  FROUDE NUMBER — mountain wave indicator   (RAP13)
#
#  Fr = U_perp / (N * H)
#  U_perp: 700mb wind speed (proxy for cross-barrier flow)
#  N: Brunt–Väisälä freq from 850–500mb temp profile
#  H: fixed 1000m terrain scale (conservative for Rocky Mountain context)
# ══════════════════════════════════════════════════════════════════════════════

_fr_cmap, _fr_norm, _fr_legend = _scale(
    bounds  = [0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 2.0, 5.0],
    colors  = ['#4575b4','#74add1','#abd9e9','#ffffbf',
               '#fee090','#fc8d59','#d73027','#a50026'],
    labels  = ['<0.25','0.25–0.50','0.50–0.75','0.75–1.0 (approach)',
               '1.0–1.25 (resonant)','1.25–1.50','1.50–2.0','≥2.0'],
)

@dataclass
class _Froude(ProductDef):
    """Fr = U_700 / (N * H) — 700mb U+V fetched together."""
    def get_values(self, cycle_dt, fxx):
        g = 9.81
        H = 1000.0
        tag = f"{self.model_id}_{cycle_dt.strftime('%Y%m%d%H')}_{fxx:02d}_froude"

        ds_uv = herbie_fetch(self.herbie_model, self.herbie_product,
                             cycle_dt, fxx,
                             [":UGRD:700 mb:|:VGRD:700 mb:"],
                             f"{tag}_uv700")
        u700 = extract_var(ds_uv, ["ugrd", "u"])
        v700 = extract_var(ds_uv, ["vgrd", "v"])
        U = np.sqrt(u700**2 + v700**2)
        lat2d, lon2d = get_latlon(ds_uv)

        ds_t850 = herbie_fetch(self.herbie_model, self.herbie_product,
                               cycle_dt, fxx, [":TMP:850 mb:"], f"{tag}_t850")
        ds_t500 = herbie_fetch(self.herbie_model, self.herbie_product,
                               cycle_dt, fxx, [":TMP:500 mb:"], f"{tag}_t500")
        T850 = extract_var(ds_t850, ["tmp", "t"])
        T500 = extract_var(ds_t500, ["tmp", "t"])

        theta850 = T850 * (1000/850)**0.286
        theta500 = T500 * (1000/500)**0.286
        theta_mean = (theta850 + theta500) / 2.0
        dz = 4000.0
        N2 = np.maximum((g / theta_mean) * ((theta500 - theta850) / dz), 1e-6)
        Fr = np.clip(U / (np.sqrt(N2) * H), 0, 5.0)
        return lat2d, lon2d, Fr

register(_Froude(
    model_id="rap13", product_id="froude",
    label="Froude Number — 700mb", units="Fr",
    herbie_model="rap", herbie_product="awp130pgrb",
    searches=[],
    cmap=_fr_cmap, norm=_fr_norm, legend=_fr_legend,
))

# ── Virga ──────────────────────────────────────────────────────────────────

_virga_cmap, _virga_norm, _virga_legend = _scale(
    bounds = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0],
    colors = ['#f7fbff','#c6dbef','#6baed6','#f16913','#cb181d'],
    labels = ['None','Low (≥20%)','Moderate (≥40%)','High (≥60%)','Extreme (≥80%)'],
)

@dataclass
class _Virga(ProductDef):
    def get_values(self, cycle_dt, fxx):
        from virga_threat import fetch_virga_arrays
        return fetch_virga_arrays(
            self.herbie_model, self.herbie_product, cycle_dt, fxx
        )

register(_Virga(
    model_id="rap13", product_id="virga",
    label="Virga Potential", units="%",
    herbie_model="rap", herbie_product="awp130pgrb",
    searches=[r"(?:TMP|RH|UGRD|VGRD):(?:500|600|700|800|850) mb"],
    cmap=_virga_cmap, norm=_virga_norm, legend=_virga_legend,
))
register(_Virga(
    model_id="hrrr", product_id="virga",
    label="Virga Potential", units="%",
    herbie_model="hrrr", herbie_product="prs",
    searches=[r"(?:TMP|DPT|UGRD|VGRD):(?:500|550|600|650|700|750|800|850) mb"],
    cmap=_virga_cmap, norm=_virga_norm, legend=_virga_legend,
))

# ── LLTI ───────────────────────────────────────────────────────────────────

_llti_cmap, _llti_norm, _llti_legend = _scale(
    bounds = [0.0, 25.0, 50.0, 75.0, 100.0],
    colors = ['#006400','#FFD700','#FF8C00','#8B0000'],
    labels = ['Low (<25)','Moderate (≥25)','High (≥50)','Extreme (≥75)'],
)

@dataclass
class _LLTI(ProductDef):
    sfc_product: str = ""
    def get_values(self, cycle_dt, fxx):
        from llti_threat import fetch_llti_arrays
        return fetch_llti_arrays(
            self.herbie_model, self.sfc_product,
            self.herbie_product, cycle_dt, fxx
        )

register(_LLTI(
    model_id="rap13", product_id="llti",
    label="Low-Level Turbulence Index", units="index",
    herbie_model="rap", herbie_product="awp130pgrb",
    sfc_product="awp130pgrb",
    searches=[],
    cmap=_llti_cmap, norm=_llti_norm, legend=_llti_legend,
))
register(_LLTI(
    model_id="hrrr", product_id="llti",
    label="Low-Level Turbulence Index", units="index",
    herbie_model="hrrr", herbie_product="prs",
    sfc_product="sfc",
    searches=[],
    cmap=_llti_cmap, norm=_llti_norm, legend=_llti_legend,
))


# ══════════════════════════════════════════════════════════════════════════════
#  TURBULENCE — Ellrod TI1 index   (RAP13 + HRRR)
#
#  TI1 = VWS * DEF
#
#  VWS = vertical wind shear magnitude across the 300–250mb layer, s⁻¹
#        = |ΔV| / Δz   where Δz is the actual geopotential height difference
#
#  DEF = total horizontal deformation, s⁻¹
#        = sqrt( DST² + DSH² )
#        DST = ∂u/∂x − ∂v/∂y   (stretching deformation)
#        DSH = ∂v/∂x + ∂u/∂y   (shearing deformation)
#        *** gradients must use physical grid spacing in meters ***
#
#  Result scaled by 1e7 to bring into display range (~0–40 for typical jets).
# ══════════════════════════════════════════════════════════════════════════════
_ti_cmap, _ti_norm, _ti_legend = _scale(
    bounds=[0, 10, 20, 30, 40, 50, 60, 75, 100, 200],
    colors=['#f7f7f7',
            '#d9f0a3',
            '#addd8e',
            '#78c679',
            '#41ab5d',
            '#fd8d3c',
            '#e31a1c',
            '#800026',
            '#4d0010'],
    labels=['Neg',
            '1 (Light)',
            '2 (Light-Mod)',
            '3 (Moderate)',
            '4 (Mod-Sev)',
            '5 (Severe)',
            '6 (Extreme)',
            '7 (Extreme+)',
            '8 (Extreme++)'],
)

def _compute_grid_spacing(lat2d, lon2d):
    """
    Estimate physical grid spacing (dy, dx) in meters from a 2-D lat/lon grid.

    Uses centered finite differences across the whole grid then takes the mean,
    which is robust for both the RAP13 Lambert-Conformal grid (~13.5 km) and
    the HRRR grid (~3 km).  Works for any model without hardcoding constants.

    Returns
    -------
    dy_m : float  – N–S grid spacing in metres
    dx_m : float  – E–W grid spacing in metres
    """
    # Degrees → metres conversions
    # 1° latitude  ≈ 111 320 m everywhere
    # 1° longitude ≈ 111 320 m × cos(lat) (varies with latitude)
    DEG_TO_M = 111_320.0

    # Row-to-row spacing (latitude direction)
    dlat = np.abs(np.diff(lat2d, axis=0))          # shape (ny-1, nx)
    dy_m = float(np.mean(dlat)) * DEG_TO_M

    # Column-to-column spacing (longitude direction), latitude-corrected
    dlon = np.abs(np.diff(lon2d, axis=1))           # shape (ny, nx-1)
    lat_mid = lat2d[:, :-1]                          # same shape for cos correction
    dx_m = float(np.mean(dlon * np.cos(np.radians(lat_mid)))) * DEG_TO_M

    return dy_m, dx_m


@dataclass
class _Turbulence(ProductDef):
    """
    Ellrod TI1 — 300mb deformation × 300–250mb vertical wind shear.

    Fixes vs. the previous version:
      • np.gradient calls now include physical dy/dx spacing → correct s⁻¹ units
      • dz is computed from actual fetched geopotential height fields, not
        a hardcoded 1500 m constant
    """
    def get_values(self, cycle_dt, fxx):
        tag = f"{self.model_id}_{cycle_dt.strftime('%Y%m%d%H')}_{fxx:02d}_turb"

        # ── fetch U/V at both levels ──────────────────────────────────────────
        ds300_uv = herbie_fetch(self.herbie_model, self.herbie_product,
                                cycle_dt, fxx,
                                [":UGRD:300 mb:|:VGRD:300 mb:"],
                                f"{tag}_uv300")
        ds250_uv = herbie_fetch(self.herbie_model, self.herbie_product,
                                cycle_dt, fxx,
                                [":UGRD:250 mb:|:VGRD:250 mb:"],
                                f"{tag}_uv250")

        # ── fetch geopotential height at both levels (needed for real dz) ────
        ds300_z  = herbie_fetch(self.herbie_model, self.herbie_product,
                                cycle_dt, fxx,
                                [":HGT:300 mb:"],
                                f"{tag}_z300")
        ds250_z  = herbie_fetch(self.herbie_model, self.herbie_product,
                                cycle_dt, fxx,
                                [":HGT:250 mb:"],
                                f"{tag}_z250")

        # ── extract arrays ────────────────────────────────────────────────────
        u300 = extract_var(ds300_uv, ["ugrd", "u", "UGRD"])
        v300 = extract_var(ds300_uv, ["vgrd", "v", "VGRD"])
        u250 = extract_var(ds250_uv, ["ugrd", "u", "UGRD"])
        v250 = extract_var(ds250_uv, ["vgrd", "v", "VGRD"])
        z300 = extract_var(ds300_z,  ["gh", "z", "hgt", "HGT", "HGHT"])
        z250 = extract_var(ds250_z,  ["gh", "z", "hgt", "HGT", "HGHT"])

        lat2d, lon2d = get_latlon(ds300_uv)

        # ── grid spacing in metres ────────────────────────────────────────────
        # Needed to convert np.gradient output from (m/s)/index → (m/s)/m = s⁻¹
        dy_m, dx_m = _compute_grid_spacing(lat2d, lon2d)

        # ── vertical wind shear (VWS) across the 300–250 mb layer ────────────
        # dz: actual geopotential height difference between 250 and 300 mb
        # (positive because 250mb is always above 300mb → z250 > z300)
        dz = z250 - z300                                # metres, 2-D field
        dz = np.maximum(dz, 100.0)                      # guard against bad data

        delta_u = u250 - u300
        delta_v = v250 - v300
        # |ΔV| / Δz → s⁻¹
        VWS = np.sqrt(delta_u**2 + delta_v**2) / dz

        # ── total horizontal deformation (DEF) at 300 mb ─────────────────────
        # np.gradient with explicit spacing → correct physical units (s⁻¹)
        du_dx = np.gradient(u300, dx_m, axis=1)         # ∂u/∂x
        du_dy = np.gradient(u300, dy_m, axis=0)         # ∂u/∂y
        dv_dx = np.gradient(v300, dx_m, axis=1)         # ∂v/∂x
        dv_dy = np.gradient(v300, dy_m, axis=0)         # ∂v/∂y

        DST = du_dx - dv_dy                             # stretching deformation
        DSH = dv_dx + du_dy                             # shearing deformation
        DEF = np.sqrt(DST**2 + DSH**2)                  # total deformation, s⁻¹

        # ── Ellrod TI1 ────────────────────────────────────────────────────────
        # VWS (s⁻¹) × DEF (s⁻¹) → s⁻²; scale × 1e7 → display range ~0–40
        TI1 = np.clip(VWS * DEF * 1e7, 0, 200)

        return lat2d, lon2d, TI1


register(_Turbulence(
    model_id="rap13", product_id="turbulence",
    label="Turbulence — Ellrod TI1 (300–250mb)", units="TI",
    herbie_model="rap", herbie_product="awp130pgrb",
    searches=[],
    cmap=_ti_cmap, norm=_ti_norm, legend=_ti_legend,
))
register(_Turbulence(
    model_id="hrrr", product_id="turbulence",
    label="Turbulence — Ellrod TI1 (300–250mb)", units="TI",
    herbie_model="hrrr", herbie_product="prs",
    searches=[],
    cmap=_ti_cmap, norm=_ti_norm, legend=_ti_legend,
))


# ══════════════════════════════════════════════════════════════════════════════
#  TURBULENCE — Stability-Weighted Ellrod   (RAP13 + HRRR)
#
#  TI_Ri = TI1 × max( −ln(Ri), 0 )
#
#  Richardson number (bulk, 300–250mb layer):
#    N²  = (g / θ_mean) × (Δθ / Δz)     Brunt-Väisälä frequency squared
#    VWS² = |ΔV|² / Δz²                  as in TI1
#    Ri   = N² / VWS²
#
#  Weight behaviour:
#    Ri < 0.25  →  −ln(Ri) > 1.39  →  signal AMPLIFIED (dynamic instability)
#    Ri = 1.0   →  −ln(Ri) = 0     →  signal ZEROED (neutral stability)
#    Ri > 1.0   →  weight  < 0     →  clipped to 0 (stable, no signal)
#
#  Original concept from four_level_ellrod_siphon.py (R. Connell, AWC).
#  Adapted to RAP13/HRRR Herbie pipeline by removing MetPy unit system
#  and computing bulk Ri over the same 300–250mb layer used for TI1.
# ══════════════════════════════════════════════════════════════════════════════
_ti_ri_cmap, _ti_ri_norm, _ti_ri_legend = _scale(
    bounds=[0, 10, 20, 30, 40, 50, 60, 75, 100, 200],
    colors=['#f7f7f7',
            '#fef0d9',
            '#fdd49e',
            '#fdbb84',
            '#fc8d59',
            '#ef6548',
            '#d7301f',
            '#b30000',
            '#7f0000'],
    labels=['Neg',
            '1 (Light)',
            '2 (Light-Mod)',
            '3 (Moderate)',
            '4 (Mod-Sev)',
            '5 (Severe)',
            '6 (Extreme)',
            '7 (Extreme+)',
            '8 (Extreme++)'],
)

@dataclass
class _TurbulenceRi(ProductDef):
    """
    Stability-Weighted Ellrod  =  TI1 × max(−ln(Ri), 0)

    Uses the same 300–250mb layer as TI1.  Richardson number is computed
    from potential-temperature gradient and wind shear across that same
    layer, so TI1 and Ri are physically consistent with each other.

    Extra GRIB messages vs. TI1: T at 300mb and 250mb only.
    Z at 300/250mb is already fetched for the TI1 dz correction.
    """
    def get_values(self, cycle_dt, fxx):
        tag = f"{self.model_id}_{cycle_dt.strftime('%Y%m%d%H')}_{fxx:02d}_turb_ri"

        # ── fetch U/V ─────────────────────────────────────────────────────────
        ds300_uv = herbie_fetch(self.herbie_model, self.herbie_product,
                                cycle_dt, fxx,
                                [":UGRD:300 mb:|:VGRD:300 mb:"],
                                f"{tag}_uv300")
        ds250_uv = herbie_fetch(self.herbie_model, self.herbie_product,
                                cycle_dt, fxx,
                                [":UGRD:250 mb:|:VGRD:250 mb:"],
                                f"{tag}_uv250")

        # ── fetch geopotential height ─────────────────────────────────────────
        ds300_z  = herbie_fetch(self.herbie_model, self.herbie_product,
                                cycle_dt, fxx,
                                [":HGT:300 mb:"],
                                f"{tag}_z300")
        ds250_z  = herbie_fetch(self.herbie_model, self.herbie_product,
                                cycle_dt, fxx,
                                [":HGT:250 mb:"],
                                f"{tag}_z250")

        # ── fetch temperature (needed for θ → N² → Ri) ───────────────────────
        ds300_t  = herbie_fetch(self.herbie_model, self.herbie_product,
                                cycle_dt, fxx,
                                [":TMP:300 mb:"],
                                f"{tag}_t300")
        ds250_t  = herbie_fetch(self.herbie_model, self.herbie_product,
                                cycle_dt, fxx,
                                [":TMP:250 mb:"],
                                f"{tag}_t250")

        # ── extract arrays ────────────────────────────────────────────────────
        u300 = extract_var(ds300_uv, ["ugrd", "u", "UGRD"])
        v300 = extract_var(ds300_uv, ["vgrd", "v", "VGRD"])
        u250 = extract_var(ds250_uv, ["ugrd", "u", "UGRD"])
        v250 = extract_var(ds250_uv, ["vgrd", "v", "VGRD"])
        z300 = extract_var(ds300_z,  ["gh", "z", "hgt", "HGT", "HGHT"])
        z250 = extract_var(ds250_z,  ["gh", "z", "hgt", "HGT", "HGHT"])
        t300 = extract_var(ds300_t,  ["t", "tmp", "TMP", "TEMP"])   # Kelvin
        t250 = extract_var(ds250_t,  ["t", "tmp", "TMP", "TEMP"])   # Kelvin

        lat2d, lon2d = get_latlon(ds300_uv)

        # ── grid spacing ──────────────────────────────────────────────────────
        dy_m, dx_m = _compute_grid_spacing(lat2d, lon2d)

        # ── vertical layer thickness ──────────────────────────────────────────
        dz = z250 - z300                                # metres, 2-D
        dz = np.maximum(dz, 100.0)                      # guard against bad data

        # ── TI1 (same arithmetic as _Turbulence) ─────────────────────────────
        delta_u = u250 - u300
        delta_v = v250 - v300
        VWS     = np.sqrt(delta_u**2 + delta_v**2) / dz   # s⁻¹

        du_dx = np.gradient(u300, dx_m, axis=1)
        du_dy = np.gradient(u300, dy_m, axis=0)
        dv_dx = np.gradient(v300, dx_m, axis=1)
        dv_dy = np.gradient(v300, dy_m, axis=0)
        DEF   = np.sqrt((du_dx - dv_dy)**2 + (dv_dx + du_dy)**2)  # s⁻¹

        TI1 = VWS * DEF * 1e7                          # unclipped here — clip after weighting

        # ── Richardson number (bulk, 300–250mb layer) ─────────────────────────
        # Potential temperature  θ = T × (1000 / P)^(R/Cp)   R/Cp ≈ 0.2857
        theta300 = t300 * (1000.0 / 300.0) ** 0.2857
        theta250 = t250 * (1000.0 / 250.0) ** 0.2857

        # Brunt-Väisälä frequency squared
        # N² = (g / θ_mean) × (Δθ / Δz)
        # A positive dθ/dz means the atmosphere is statically stable (normal),
        # so N² > 0 in almost all upper-level situations.
        g           = 9.81                              # m s⁻²
        theta_mean  = (theta300 + theta250) / 2.0
        dtheta_dz   = (theta250 - theta300) / dz       # K m⁻¹
        N2          = (g / theta_mean) * dtheta_dz     # s⁻²

        # Wind shear magnitude squared (same ΔV already computed above)
        VWS2 = np.maximum(VWS**2, 1e-20)               # guard divide-by-zero

        # Bulk Ri = N² / |VWS|²
        # Clip N² at 0 before dividing — negative N² means absolute instability,
        # which would give negative Ri; those regions get Ri_safe → 0.01 → large
        # positive weight, but that's physically appropriate (very unstable).
        Ri = np.maximum(N2, 0.0) / VWS2

        # ── stability weight  max( −ln(Ri), 0 ) ──────────────────────────────
        # Ri_safe prevents log(0) or log(negative); floor at 0.01 means the
        # maximum possible weight is −ln(0.01) ≈ 4.6 (amplification cap).
        Ri_safe = np.clip(Ri, 0.01, None)
        weight  = np.maximum(-np.log(Ri_safe), 0.0)

        # ── stability-weighted TI ─────────────────────────────────────────────
        TI_Ri = np.clip(TI1 * weight, 0, 200)

        return lat2d, lon2d, TI_Ri

    def get_contour_overlay(self, cycle_dt, fxx) -> dict | None:
        """Return 250mb height contours to overlay on the TI·Ri fill."""
        tag = f"{self.model_id}_{cycle_dt.strftime('%Y%m%d%H')}_{fxx:02d}_turb_ri"
        try:
            ds250_z = herbie_fetch(
                self.herbie_model, self.herbie_product,
                cycle_dt, fxx,
                [":HGT:250 mb:"],
                f"{tag}_z250",
            )
            z250 = extract_var(ds250_z, ["gh", "z", "hgt", "HGT", "HGHT"])
            lat2d, lon2d = get_latlon(ds250_z)
            z250_100ft = z250 * 3.28084 / 100.0
            return {
                "lat2d":      lat2d,
                "lon2d":      lon2d,
                "data":       z250_100ft,
                # Every 3 - double the density (standard for operational 250mb charts)
                "levels":     list(np.arange(310, 420, 3)),
                "color":      "#1a1a1a",
                "linewidths": 1.1,
                "alpha":      0.80,
                "label_fmt":  "%i",
            }
        except Exception as e:
            log.warning(f"Height overlay fetch failed: {e}")
            return None

register(_TurbulenceRi(
    model_id="rap13", product_id="turbulence_ri",
    label="Turbulence — Stability-Weighted Ellrod (300–250mb)", units="TI·Ri",
    herbie_model="rap", herbie_product="awp130pgrb",
    searches=[],
    _ti_ri_cmap, _ti_ri_norm, _ti_ri_legend,
))
register(_TurbulenceRi(
    model_id="hrrr", product_id="turbulence_ri",
    label="Turbulence — Stability-Weighted Ellrod (300–250mb)", units="TI·Ri",
    herbie_model="hrrr", herbie_product="prs",
    searches=[],
    _ti_ri_cmap, _ti_ri_norm, _ti_ri_legend,
))
