"""
artcc_boundaries.py
Downloads accurate ARTCC boundary GeoJSON from multiple aviation data sources.
Primary: tar1090 (ADS-B display tool with well-maintained FAA airspace data)
Fallback: ArcGIS Hub FAA dataset, jsDelivr CDN mirror
Caches to static/artcc.geojson — re-downloaded only if missing or < 50KB.
"""

import json, logging, threading, urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
ARTCC_FILE = STATIC_DIR / "artcc.geojson"
_LOCK      = threading.Lock()
_ready     = False

# Sources in priority order — all serve accurate FAA ARTCC boundary polygons
_SOURCES = [
    # tar1090 — maintained ADS-B display tool, high-quality aviation GeoJSON
    "https://raw.githubusercontent.com/wiedehopf/tar1090/master/html/geojson/US_ARTCC_boundaries.geojson",
    # jsDelivr CDN mirror of same repo (different network path)
    "https://cdn.jsdelivr.net/gh/wiedehopf/tar1090@master/html/geojson/US_ARTCC_boundaries.geojson",
    # FAA ARTCC dataset via ArcGIS Hub REST API (official FAA data)
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
    """Call once at startup (preferably in a daemon thread)."""
    global _ready
    with _LOCK:
        if _ready and ARTCC_FILE.exists():
            return ARTCC_FILE

        STATIC_DIR.mkdir(parents=True, exist_ok=True)

        # Good cached file — skip download
        if ARTCC_FILE.exists() and ARTCC_FILE.stat().st_size > 50_000:
            log.info(f"ARTCC: cached {ARTCC_FILE.stat().st_size // 1024} KB — ready")
            _ready = True
            return ARTCC_FILE

        # Stale/empty cache — delete and re-download
        if ARTCC_FILE.exists():
            log.info(f"ARTCC: cached file too small ({ARTCC_FILE.stat().st_size} B), re-fetching")
            ARTCC_FILE.unlink()

        for url in _SOURCES:
            log.info(f"ARTCC: trying {url[:70]}")
            try:
                raw  = _fetch(url)
                gj   = json.loads(raw)
                n    = len(gj.get("features", []))
                if n < 5:
                    raise ValueError(f"Only {n} features — likely wrong endpoint")
                ARTCC_FILE.write_bytes(raw)
                log.info(f"ARTCC: downloaded {n} features ({len(raw)//1024} KB) → {ARTCC_FILE}")
                _ready = True
                return ARTCC_FILE
            except Exception as e:
                log.warning(f"ARTCC: {url[:60]} failed — {e}")

        log.error("ARTCC: all sources failed — map will show no ARTCC boundaries")
        _ready = True
        return ARTCC_FILE


def get_artcc_geojson() -> dict:
    if ARTCC_FILE.exists() and ARTCC_FILE.stat().st_size > 100:
        return json.loads(ARTCC_FILE.read_text())
    return {"type": "FeatureCollection", "features": []}
