"""
products/__init__.py — ProductDef base class and registry.

Every model/product combination is a ProductDef instance registered here.
The registry drives:
  - The model + product dropdowns in the UI  (/api/products)
  - Generic fetch/render routes              (/api/image/<model>/<product>/...)
  - Cycle status checks                      (/api/status/<model>/<product>)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import matplotlib.colors as mcolors


@dataclass
class ProductDef:
    """
    Describes one model × product combination.

    Simple products (single GRIB variable) only need to set the
    class attributes and let get_values() use the default implementation.

    Computed products (derived fields like icing, Froude, turbulence)
    override get_values() to fetch multiple fields and do math.
    """
    # Identifiers
    model_id:   str          # e.g. "rap13", "hrrr"
    product_id: str          # e.g. "wind_gust", "mslp"
    label:      str          # e.g. "Wind Gusts — Surface"

    # Herbie config
    herbie_model:   str      # e.g. "rap", "hrrr"
    herbie_product: str      # e.g. "awp130pgrb", "sfc"
    searches:       list     # GRIB search strings tried in order
    fxx_max:        int = 18

    # Display config
    units:       str = ""
    render_mode: str = "fill"    # "fill" or "contour"
    stride:      int = 2         # point-sampling stride
    supports_barbs: bool = False
    barb_stride: int = 6

    # Colormap — set by each product definition below
    cmap:   object = None
    norm:   object = None
    legend: list   = field(default_factory=list)

    def get_values(self, cycle_dt: datetime, fxx: int
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fetch GRIB data and return (lat2d, lon2d, vals2d) in display units.
        Override this in subclasses for computed/multi-variable products.
        """
        from renderer import herbie_fetch, extract_var, get_latlon
        tag  = f"{self.model_id}_{cycle_dt.strftime('%Y%m%d%H')}_{fxx:02d}"
        ds   = herbie_fetch(self.herbie_model, self.herbie_product,
                            cycle_dt, fxx, self.searches, tag)
        vals = extract_var(ds, self._var_hints)
        lat2d, lon2d = get_latlon(ds)
        return lat2d, lon2d, self._units_fn(vals)

    # Subclasses set these to customise the default get_values()
    _var_hints: list  = field(default_factory=list)
    _units_fn:  object = field(default=lambda v: v)

    def get_barb_data(self, cycle_dt: datetime, fxx: int):
        """
        Return (lat2d, lon2d, u_ms, v_ms) for wind barb rendering.
        Override in subclasses that set supports_barbs=True.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support barbs")

    def get_point_values(self, cycle_dt: datetime, fxx: int) -> dict:
        """
        Return dict of named value arrays for cursor display.
        Default: single 'value' key matching get_values() output.
        Override in subclasses to expose multiple fields (e.g. AGL + MSL).
        """
        lat2d, lon2d, vals2d = self.get_values(cycle_dt, fxx)
        return {"value": vals2d}

# ── Registry ──────────────────────────────────────────────────────────────────
# REGISTRY[model_id][product_id] = ProductDef
REGISTRY: dict[str, dict[str, ProductDef]] = {}

def register(p: ProductDef):
    REGISTRY.setdefault(p.model_id, {})[p.product_id] = p

def get_product(model_id: str, product_id: str) -> ProductDef:
    try:
        return REGISTRY[model_id][product_id]
    except KeyError:
        raise ValueError(f"Unknown product: {model_id}/{product_id}")

def registry_json() -> list:
    """Return the registry as a JSON-serialisable list for /api/products."""
    out = []
    for model_id, products in REGISTRY.items():
        out.append({
            "model_id": model_id,
            "label":    _MODEL_LABELS.get(model_id, model_id),
            "products": [
                {"product_id": pid,
                 "label":      p.label,
                 "units":      p.units,
                 "legend":     p.legend,
                 "supports_barbs": p.supports_barbs,
                 "fxx_max":    p.fxx_max}
                for pid, p in products.items()
            ]
        })
    return out

_MODEL_LABELS = {
    "rap13": "RAP 13km",
    "hrrr":  "HRRR 3km",
    "gfs":   "GFS 0.25°",
    "rrfs":  "RRFS 3km",
}
