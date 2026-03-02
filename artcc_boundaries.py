"""
artcc_boundaries.py
Serves ARTCC boundary GeoJSON from static/artcc.geojson.
File sourced from tar1090 (https://github.com/wiedehopf/tar1090),
which maintains accurate FAA airspace boundaries for ADS-B display.
To update: re-download from the URL above and replace static/artcc.geojson.
"""

import json, logging
from pathlib import Path

log = logging.getLogger(__name__)

ARTCC_FILE = Path(__file__).parent / "static" / "artcc.geojson"


def ensure_artcc_geojson() -> Path:
    """No-op — file is committed to repo. Called at startup for compatibility."""
    if ARTCC_FILE.exists():
        gj = get_artcc_geojson()
        log.info(f"ARTCC: ready — {len(gj['features'])} features, "
                 f"{ARTCC_FILE.stat().st_size // 1024} KB")
    else:
        log.error("ARTCC: static/artcc.geojson not found — commit it to the repo!")
    return ARTCC_FILE


def get_artcc_geojson() -> dict:
    if ARTCC_FILE.exists():
        return json.loads(ARTCC_FILE.read_text())
    log.error("ARTCC: static/artcc.geojson missing")
    return {"type": "FeatureCollection", "features": []}
