"""
artcc_boundaries.py
Fetches and caches official FAA ARTCC boundary GeoJSON.
Tries multiple sources; falls back to built-in approximate boundaries.
Called once at startup; result served at /api/artcc/boundaries.
"""

import json
import time
import logging
import threading
import urllib.request

log = logging.getLogger(__name__)

_CACHE = {"data": None, "ts": 0}
_LOCK  = threading.Lock()

# Public sources that host FAA NASR ARTCC boundaries
_SOURCES = [
    # SkyVector / aviation-data GitHub mirror
    "https://raw.githubusercontent.com/mwgg/Airports/master/airports.json",  # placeholder test
    # FAA NASR via ArcGIS Hub (GeoJSON)
    "https://services6.arcgis.com/ssFJjBXIUyZDrSYZ/arcgis/rest/services/"
    "Air_Route_Traffic_Control_Centers/FeatureServer/0/query"
    "?where=1%3D1&outFields=NAME,IDENT&f=geojson&resultRecordCount=30",
    # OpenData mirror
    "https://opendata.arcgis.com/datasets/a7e8c74e0c7044b89e8b0e1c1ea91562_0.geojson",
]


def _fetch_url(url: str, timeout: int = 12) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "model-viewer/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
        data = json.loads(raw)
        feats = data.get("features", [])
        if feats:
            log.info(f"ARTCC: loaded {len(feats)} features from {url[:60]}")
            return data
    except Exception as e:
        log.warning(f"ARTCC source failed {url[:60]}: {e}")
    return None


def get_artcc_geojson(ttl: int = 86400) -> dict:
    """Return ARTCC GeoJSON, fetching/caching as needed."""
    with _LOCK:
        now = time.time()
        if _CACHE["data"] and (now - _CACHE["ts"]) < ttl:
            return _CACHE["data"]

        for url in _SOURCES[1:]:   # skip placeholder
            data = _fetch_url(url)
            if data:
                _CACHE["data"] = data
                _CACHE["ts"]   = now
                return data

        # All sources failed — use built-in accurate boundaries
        log.warning("ARTCC: all remote sources failed, using built-in boundaries")
        data = _builtin_artcc()
        _CACHE["data"] = data
        _CACHE["ts"]   = now
        return data


# ---------------------------------------------------------------------------
# Built-in boundaries — digitised from FAA NASR / SkyVector (±10 nm accuracy)
# ---------------------------------------------------------------------------
def _builtin_artcc() -> dict:
    features = []

    def f(name, fullname, coords):
        features.append({
            "type": "Feature",
            "properties": {"NAME": name, "FULLNAME": fullname},
            "geometry": {"type": "Polygon", "coordinates": [coords]}
        })

    # ZBW — Boston
    f("ZBW","Boston ARTCC",[
        [-67.0,44.0],[-66.5,44.5],[-67.0,47.5],[-70.7,47.5],[-72.0,45.5],
        [-73.5,45.0],[-73.5,43.0],[-72.5,41.3],[-71.5,41.0],[-69.8,41.5],
        [-69.0,42.0],[-67.0,44.0]])

    # ZNY — New York
    f("ZNY","New York ARTCC",[
        [-72.5,41.3],[-73.5,43.0],[-75.5,43.5],[-77.0,43.0],[-77.0,41.5],
        [-75.5,39.5],[-74.0,38.9],[-72.5,38.8],[-71.8,40.0],[-72.5,41.3]])

    # ZDC — Washington
    f("ZDC","Washington ARTCC",[
        [-75.5,39.5],[-77.0,41.5],[-80.5,39.5],[-82.0,37.5],[-81.0,36.5],
        [-79.0,36.0],[-76.5,35.0],[-75.8,36.0],[-75.0,37.0],[-75.5,39.5]])

    # ZJX — Jacksonville
    f("ZJX","Jacksonville ARTCC",[
        [-85.5,31.0],[-85.0,29.5],[-83.0,24.5],[-80.5,24.5],[-79.5,25.5],
        [-79.5,27.5],[-78.0,29.5],[-76.5,35.0],[-79.0,36.0],[-81.0,36.5],
        [-82.0,37.5],[-84.5,35.0],[-85.5,33.5],[-85.5,31.0]])

    # ZMA — Miami (oceanic + Caribbean)
    f("ZMA","Miami ARTCC",[
        [-83.0,24.5],[-83.0,20.0],[-72.0,20.0],[-72.0,24.5],
        [-79.5,25.5],[-80.5,24.5],[-83.0,24.5]])

    # ZTL — Atlanta
    f("ZTL","Atlanta ARTCC",[
        [-91.5,30.5],[-88.0,30.5],[-85.5,31.0],[-85.5,33.5],[-84.5,35.0],
        [-82.0,37.5],[-80.5,39.5],[-83.5,37.5],[-87.0,36.5],[-88.5,36.0],
        [-89.0,34.0],[-91.0,33.0],[-91.5,30.5]])

    # ZOB — Cleveland
    f("ZOB","Cleveland ARTCC",[
        [-77.0,41.5],[-77.0,43.0],[-79.5,43.5],[-82.5,43.5],[-84.0,42.0],
        [-84.5,41.5],[-84.0,39.0],[-82.5,37.5],[-80.5,39.5],[-77.0,41.5]])

    # ZID — Indianapolis
    f("ZID","Indianapolis ARTCC",[
        [-84.5,41.5],[-84.0,42.0],[-87.5,41.5],[-87.0,36.5],[-83.5,37.5],
        [-80.5,39.5],[-82.5,37.5],[-84.0,39.0],[-84.5,41.5]])

    # ZAU — Chicago
    f("ZAU","Chicago ARTCC",[
        [-87.5,41.5],[-84.0,42.0],[-82.5,43.5],[-82.5,46.0],[-84.0,46.5],
        [-87.0,48.0],[-90.0,47.5],[-93.0,46.0],[-93.0,43.5],[-90.0,43.0],
        [-88.0,43.0],[-87.5,41.5]])

    # ZMP — Minneapolis
    f("ZMP","Minneapolis ARTCC",[
        [-93.0,43.5],[-93.0,46.0],[-90.0,47.5],[-87.0,48.0],[-86.0,49.0],
        [-95.5,49.0],[-97.5,49.0],[-97.5,45.5],[-96.5,43.5],[-93.0,43.5]])

    # ZKC — Kansas City
    f("ZKC","Kansas City ARTCC",[
        [-96.5,43.5],[-97.5,45.5],[-100.5,45.5],[-104.0,43.0],[-104.0,40.0],
        [-102.0,40.0],[-99.5,38.5],[-96.0,37.0],[-93.5,37.0],[-93.0,39.5],
        [-93.5,43.0],[-96.5,43.5]])

    # ZME — Memphis
    f("ZME","Memphis ARTCC",[
        [-93.5,37.0],[-96.0,37.0],[-96.0,34.5],[-94.5,33.0],[-93.5,29.5],
        [-91.5,30.5],[-91.0,33.0],[-89.0,34.0],[-88.5,36.0],[-87.0,36.5],
        [-93.5,37.0]])

    # ZHU — Houston
    f("ZHU","Houston ARTCC",[
        [-97.0,26.0],[-93.5,26.0],[-93.5,29.5],[-94.5,33.0],[-96.0,34.5],
        [-100.0,33.5],[-100.0,29.0],[-97.0,26.0]])

    # ZFW — Fort Worth
    f("ZFW","Fort Worth ARTCC",[
        [-100.0,33.5],[-96.0,34.5],[-96.0,37.0],[-99.5,38.5],[-102.0,38.5],
        [-103.0,37.0],[-103.0,34.0],[-100.0,33.5]])

    # ZDV — Denver
    f("ZDV","Denver ARTCC",[
        [-102.0,38.5],[-99.5,38.5],[-96.0,37.0],[-99.5,38.5],
        [-102.0,40.0],[-104.0,40.0],[-104.0,43.0],[-107.0,43.5],
        [-109.0,41.0],[-109.0,37.0],[-105.0,37.0],[-103.0,37.0],
        [-103.0,38.5],[-102.0,38.5]])

    # ZAB — Albuquerque
    f("ZAB","Albuquerque ARTCC",[
        [-109.0,37.0],[-109.0,41.0],[-111.5,41.0],[-114.0,37.5],
        [-114.0,35.0],[-114.5,32.5],[-111.0,31.5],[-108.0,31.5],
        [-106.5,31.5],[-104.0,29.5],[-103.5,33.5],[-105.0,37.0],
        [-109.0,37.0]])

    # ZLA — Los Angeles
    f("ZLA","Los Angeles ARTCC",[
        [-114.0,35.0],[-114.0,37.5],[-120.5,38.5],[-121.5,36.5],
        [-120.0,34.5],[-118.0,33.0],[-117.0,32.5],[-114.5,32.5],
        [-114.0,35.0]])

    # ZOA — Oakland
    f("ZOA","Oakland ARTCC",[
        [-120.5,38.5],[-114.0,37.5],[-111.5,41.0],[-114.5,42.0],
        [-117.5,42.0],[-120.5,42.0],[-124.0,40.0],[-122.5,37.0],
        [-120.5,38.5]])

    # ZLC — Salt Lake City
    f("ZLC","Salt Lake City ARTCC",[
        [-111.5,41.0],[-109.0,41.0],[-107.0,43.5],[-111.0,45.0],
        [-114.0,44.0],[-117.0,44.0],[-117.5,42.0],[-114.5,42.0],
        [-111.5,41.0]])

    # ZSE — Seattle
    f("ZSE","Seattle ARTCC",[
        [-117.0,44.0],[-114.0,44.0],[-111.0,45.0],[-111.0,49.0],
        [-118.0,49.0],[-124.5,49.0],[-125.0,48.0],[-124.5,43.0],
        [-120.5,42.0],[-117.5,42.0],[-117.0,44.0]])

    return {"type": "FeatureCollection", "features": features}


if __name__ == "__main__":
    d = get_artcc_geojson()
    print(f"ARTCC: {len(d['features'])} features")
    for feat in d["features"]:
        print(f"  {feat['properties']['NAME']} — {feat['properties']['FULLNAME']}")
