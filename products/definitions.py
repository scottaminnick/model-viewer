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
    _units_fn:  object = field(default=lambda v: np.maximum(v, 0.0) * 3.28084)
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
#  ICING THREAT — simplified CIP-like index   (RAP13 + HRRR)
#
#  Method: fetch cloud liquid water mixing ratio (CLWMR) and temperature
#  at 850, 700, 600, 500 mb.  Icing index = fraction of levels where
#  -20°C < T < 0°C AND CLWMR > 0.01 g/kg.
# ══════════════════════════════════════════════════════════════════════════════

_ice_cmap, _ice_norm, _ice_legend = _scale(
    bounds  = [0, 0.1, 0.25, 0.50, 0.75, 1.01],
    colors  = ['#f7fbff','#bdd7e7','#6baed6','#2171b5','#08306b'],
    labels  = ['None','Trace','Light','Moderate','Severe'],
)

@dataclass
class _IcingThreat(ProductDef):
    """
    Simplified icing index: fraction of pressure levels with supercooled
    liquid water (SLW) present (T in [-20,0]°C and CLWMR > threshold).
    """
    def get_values(self, cycle_dt, fxx):
        levels = [850, 700, 600, 500]
        tag    = f"{self.model_id}_{cycle_dt.strftime('%Y%m%d%H')}_{fxx:02d}_icing"
        icing_count = None
        lat2d = lon2d = None

        for lev in levels:
            try:
                ds_t = herbie_fetch(
                    self.herbie_model, self.herbie_product,
                    cycle_dt, fxx,
                    [f":TMP:{lev} mb:"],
                    f"{tag}_{lev}t")
                ds_c = herbie_fetch(
                    self.herbie_model, self.herbie_product,
                    cycle_dt, fxx,
                    [f":CLWMR:{lev} mb:", f":CLMR:{lev} mb:"],
                    f"{tag}_{lev}c")

                temp_c = extract_var(ds_t, ["tmp","t"]) - 273.15
                clwmr  = extract_var(ds_c, ["clwmr","clmr"]) * 1000.0  # kg/kg→g/kg

                slw = ((temp_c > -20) & (temp_c < 0) & (clwmr > 0.01)).astype(float)

                if icing_count is None:
                    icing_count = slw
                    lat2d, lon2d = get_latlon(ds_t)
                else:
                    icing_count = icing_count + slw
            except Exception:
                continue

        if icing_count is None:
            raise RuntimeError("Could not fetch any icing levels")

        icing_index = np.clip(icing_count / len(levels), 0.0, 1.0)
        return lat2d, lon2d, icing_index

register(_IcingThreat(
    model_id="rap13", product_id="icing",
    label="Icing Threat (SLW Index)", units="index",
    herbie_model="rap", herbie_product="awp130pgrb",
    searches=[],
    cmap=_ice_cmap, norm=_ice_norm, legend=_ice_legend,
))
register(_IcingThreat(
    model_id="hrrr", product_id="icing",
    label="Icing Threat (SLW Index)", units="index",
    herbie_model="hrrr", herbie_product="prs",
    searches=[],
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


# ══════════════════════════════════════════════════════════════════════════════
#  TURBULENCE — Ellrod TI index   (RAP13 + HRRR)
#
#  Ellrod TI1 = Vws * Def
#  Vws = vertical wind shear (400–250mb layer)
#  Def = horizontal deformation  (sqrt(DST² + DSH²))
# ══════════════════════════════════════════════════════════════════════════════

_ti_cmap, _ti_norm, _ti_legend = _scale(
    bounds  = [0, 4, 8, 12, 16, 20, 28, 40, 200],
    colors  = ['#f7f7f7','#d9f0a3','#addd8e','#78c679',
               '#41ab5d','#fd8d3c','#e31a1c','#800026'],
    labels  = ['Neg','1 (Light)','2 (Light-Mod)','3 (Moderate)',
               '4 (Mod-Sev)','5 (Severe)','6 (Extreme)','7 (Extreme+)'],
)

@dataclass
class _Turbulence(ProductDef):
    """Ellrod TI — 300mb and 250mb U+V each fetched together."""
    def get_values(self, cycle_dt, fxx):
        tag = f"{self.model_id}_{cycle_dt.strftime('%Y%m%d%H')}_{fxx:02d}_turb"

        ds300 = herbie_fetch(self.herbie_model, self.herbie_product,
                             cycle_dt, fxx,
                             [":UGRD:300 mb:|:VGRD:300 mb:"],
                             f"{tag}_uv300")
        ds250 = herbie_fetch(self.herbie_model, self.herbie_product,
                             cycle_dt, fxx,
                             [":UGRD:250 mb:|:VGRD:250 mb:"],
                             f"{tag}_uv250")

        u300 = extract_var(ds300, ["ugrd", "u"])
        v300 = extract_var(ds300, ["vgrd", "v"])
        u250 = extract_var(ds250, ["ugrd", "u"])
        v250 = extract_var(ds250, ["vgrd", "v"])
        lat2d, lon2d = get_latlon(ds300)

        dz = 1500.0
        Vws = np.sqrt((u250-u300)**2 + (v250-v300)**2) / dz

        du_dx = np.gradient(u300, axis=1)
        du_dy = np.gradient(u300, axis=0)
        dv_dx = np.gradient(v300, axis=1)
        dv_dy = np.gradient(v300, axis=0)
        Def = np.sqrt((du_dx - dv_dy)**2 + (dv_dx + du_dy)**2)

        TI = np.clip(Vws * Def * 1e7, 0, 200)
        return lat2d, lon2d, TI

register(_Turbulence(
    model_id="rap13", product_id="turbulence",
    label="Turbulence — Ellrod TI (300–250mb)", units="TI",
    herbie_model="rap", herbie_product="awp130pgrb",
    searches=[],
    cmap=_ti_cmap, norm=_ti_norm, legend=_ti_legend,
))
register(_Turbulence(
    model_id="hrrr", product_id="turbulence",
    label="Turbulence — Ellrod TI (300–250mb)", units="TI",
    herbie_model="hrrr", herbie_product="prs",
    searches=[],
    cmap=_ti_cmap, norm=_ti_norm, legend=_ti_legend,
))
