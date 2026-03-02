"""
artcc_boundaries.py
Downloads accurate ARTCC boundaries at startup and caches to /tmp.
Stored in /tmp (not static/) so the Docker image never provides a stale version.
"""

import json, logging, threading, urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

# /tmp survives the request but NOT across deploys — forces a fresh download each boot
ARTCC_FILE = Path("/tmp/artcc.geojson")
_LOCK      = threading.Lock()
_ready     = False

_SOURCES = [
    # tar1090 — ADS-B display tool, well-maintained FAA airspace GeoJSON
    "https://raw.githubusercontent.com/wiedehopf/tar1090/master/html/geojson/US_ARTCC_boundaries.geojson",
    # jsDelivr CDN mirror (different network path from raw.githubusercontent.com)
    "https://cdn.jsdelivr.net/gh/wiedehopf/tar1090@master/html/geojson/US_ARTCC_boundaries.geojson",
    # FAA dataset via ArcGIS Hub REST API
    (
        "https://services6.arcgis.com/ssFJjBXIUyZDrSYZ/arcgis/rest/services/"
        "Air_Route_Traffic_Control_Centers/FeatureServer/0/query"
        "?where=1%3D1&outFields=NAME,IDENT&f=geojson"
    ),
]


def _fetch(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "model-viewer/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def ensure_artcc_geojson() -> Path:
    """Download ARTCC GeoJSON at startup. Call in a daemon thread."""
    global _ready
    with _LOCK:
        if _ready:
            return ARTCC_FILE

        for url in _SOURCES:
            log.info(f"ARTCC: trying {url[:75]}")
            try:
                raw = _fetch(url)
                gj  = json.loads(raw)
                n   = len(gj.get("features", []))
                if n < 5:
                    raise ValueError(f"only {n} features returned")
                ARTCC_FILE.write_bytes(raw)
                log.info(f"ARTCC: ✅ {n} features, {len(raw)//1024} KB → {ARTCC_FILE}")
                _ready = True
                return ARTCC_FILE
            except Exception as e:
                log.warning(f"ARTCC: ❌ {url[:60]} — {e}")

        log.error("ARTCC: all sources failed — no boundaries will be shown")
        _ready = True
        return ARTCC_FILE


def get_artcc_geojson() -> dict:
    if ARTCC_FILE.exists() and ARTCC_FILE.stat().st_size > 1000:
        return json.loads(ARTCC_FILE.read_text())
    return {"type": "FeatureCollection", "features": []}
