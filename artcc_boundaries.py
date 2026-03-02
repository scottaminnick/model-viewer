"""
artcc_boundaries.py
Serves ARTCC boundary GeoJSON from static/artcc.geojson.
That file is committed to the repo — no runtime download needed.
ARTCC boundaries change extremely rarely; update the file when FAA publishes changes.
"""
import json, logging
from pathlib import Path

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
ARTCC_FILE = STATIC_DIR / "artcc.geojson"


def ensure_artcc_geojson():
    """No-op — file is committed to repo. Called at startup for API compatibility."""
    if ARTCC_FILE.exists():
        log.info(f"ARTCC: static file ready ({ARTCC_FILE.stat().st_size // 1024} KB, "
                 f"{len(get_artcc_geojson()['features'])} centers)")
    else:
        log.warning("ARTCC: static/artcc.geojson missing — commit it to the repo!")
    return ARTCC_FILE


def get_artcc_geojson() -> dict:
    if ARTCC_FILE.exists():
        return json.loads(ARTCC_FILE.read_text())
    log.error("ARTCC: static/artcc.geojson not found")
    return {"type": "FeatureCollection", "features": []}
