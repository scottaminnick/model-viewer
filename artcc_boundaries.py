"""
artcc_boundaries.py — FAA NASR ARTCC boundary downloader
Downloads the official 28-day NASR subscription zip, extracts ARB.shp,
converts to GeoJSON and writes to static/artcc.geojson at startup.
Falls back to hand-digitised boundaries if download fails.
"""

import io, json, logging, threading, urllib.request, zipfile
from datetime import date, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

STATIC_DIR   = Path(__file__).parent / "static"
ARTCC_FILE   = STATIC_DIR / "artcc.geojson"
_LOCK        = threading.Lock()
_initialized = False


# ── NASR cycle helpers ────────────────────────────────────────────────────────

def _nasr_cycle_dates(n=3):
    anchor = date(2025, 1, 23)
    today  = date.today()
    latest = anchor + timedelta(days=((today - anchor).days // 28) * 28)
    return [latest - timedelta(days=28*i) for i in range(n)]

def _nasr_url(d):
    return f"https://nfdc.faa.gov/webContent/28DaySub/28DaySubscription_Effective_{d}.zip"


# ── Download + convert ────────────────────────────────────────────────────────

def _download(url, timeout=90):
    req = urllib.request.Request(url, headers={"User-Agent": "model-viewer/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _zip_to_geojson(zip_bytes):
    import geopandas as gpd, tempfile

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        names = z.namelist()
        # Find ARB.shp — may be at root or in ARB/ subdirectory
        shp_entry = next((n for n in names if n.endswith("ARB.shp")), None)
        if not shp_entry:
            raise FileNotFoundError(
                f"ARB.shp not found. Top entries: {[n for n in names[:20]]}")

        prefix = shp_entry[:-len("ARB.shp")]   # e.g. "ARB/" or ""
        with tempfile.TemporaryDirectory() as td:
            for ext in [".shp", ".dbf", ".shx", ".prj", ".cpg"]:
                entry = f"{prefix}ARB{ext}"
                if entry in names:
                    (Path(td) / f"ARB{ext}").write_bytes(z.read(entry))
            gdf = gpd.read_file(Path(td) / "ARB.shp")

    log.info(f"ARB columns: {list(gdf.columns)}")
    log.info(f"ARB shape:   {gdf.shape}")

    # Identify the ARTCC-center identifier column
    id_col = next((c for c in ["ARTCC_ID","IDENT","ART_ID","CENTER","NAME"]
                   if c in gdf.columns), None)
    name_col = next((c for c in ["NAME","ARTCC_NAME","FULL_NAME","ARTCC_ID"]
                     if c in gdf.columns), None)

    # Dissolve sectors → single polygon per center
    if id_col:
        gdf = gdf.dissolve(by=id_col, as_index=False)

    # Reproject to WGS84
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    features = []
    for _, row in gdf.iterrows():
        ident    = str(row.get(id_col,   "")) if id_col   else ""
        fullname = str(row.get(name_col, "")) if name_col else ""
        features.append({
            "type": "Feature",
            "properties": {"NAME": ident or fullname, "IDENT": ident,
                           "FULLNAME": fullname},
            "geometry": row.geometry.__geo_interface__,
        })

    return {"type": "FeatureCollection", "features": features}


# ── Public API ────────────────────────────────────────────────────────────────

def ensure_artcc_geojson():
    """Call once at app startup. Returns path to artcc.geojson."""
    global _initialized
    with _LOCK:
        if _initialized and ARTCC_FILE.exists():
            return ARTCC_FILE

        STATIC_DIR.mkdir(parents=True, exist_ok=True)

        # Re-use existing file if it's from this cycle already
        if ARTCC_FILE.exists() and ARTCC_FILE.stat().st_size > 50_000:
            log.info(f"ARTCC: cached file exists ({ARTCC_FILE.stat().st_size//1024} KB)")
            _initialized = True
            return ARTCC_FILE

        for d in _nasr_cycle_dates(3):
            url = _nasr_url(d)
            log.info(f"ARTCC: trying NASR cycle {d}")
            try:
                raw = _download(url)
                log.info(f"ARTCC: downloaded {len(raw)//1024} KB, converting...")
                gj = _zip_to_geojson(raw)
                ARTCC_FILE.write_text(json.dumps(gj, separators=(',',':')))
                log.info(f"ARTCC: {len(gj['features'])} centers → {ARTCC_FILE}")
                _initialized = True
                return ARTCC_FILE
            except Exception as e:
                log.warning(f"ARTCC: cycle {d} failed — {e}")

        log.warning("ARTCC: all downloads failed, writing built-in fallback")
        ARTCC_FILE.write_text(json.dumps(_builtin(), separators=(',',':')))
        _initialized = True
        return ARTCC_FILE


def get_artcc_geojson():
    if ARTCC_FILE.exists():
        return json.loads(ARTCC_FILE.read_text())
    return _builtin()


# ── Built-in fallback ─────────────────────────────────────────────────────────

def _builtin():
    F = []
    def f(n, fn, c):
        F.append({"type":"Feature",
                  "properties":{"NAME":n,"IDENT":n,"FULLNAME":fn},
                  "geometry":{"type":"Polygon","coordinates":[c]}})
    f("ZBW","Boston",[[-67.0,44.0],[-66.5,44.5],[-67.0,47.5],[-70.7,47.5],[-72.0,45.5],[-73.5,45.0],[-73.5,43.0],[-72.5,41.3],[-71.5,41.0],[-69.8,41.5],[-69.0,42.0],[-67.0,44.0]])
    f("ZNY","New York",[[-72.5,41.3],[-73.5,43.0],[-75.5,43.5],[-77.0,43.0],[-77.0,41.5],[-75.5,39.5],[-74.0,38.9],[-72.5,38.8],[-71.8,40.0],[-72.5,41.3]])
    f("ZDC","Washington",[[-75.5,39.5],[-77.0,41.5],[-80.5,39.5],[-82.0,37.5],[-81.0,36.5],[-79.0,36.0],[-76.5,35.0],[-75.8,36.0],[-75.0,37.0],[-75.5,39.5]])
    f("ZJX","Jacksonville",[[-85.5,31.0],[-85.0,29.5],[-83.0,24.5],[-80.5,24.5],[-79.5,25.5],[-79.5,27.5],[-78.0,29.5],[-76.5,35.0],[-79.0,36.0],[-81.0,36.5],[-82.0,37.5],[-84.5,35.0],[-85.5,33.5],[-85.5,31.0]])
    f("ZMA","Miami",[[-83.0,24.5],[-83.0,20.0],[-72.0,20.0],[-72.0,24.5],[-79.5,25.5],[-80.5,24.5],[-83.0,24.5]])
    f("ZTL","Atlanta",[[-91.5,30.5],[-88.0,30.5],[-85.5,31.0],[-85.5,33.5],[-84.5,35.0],[-82.0,37.5],[-80.5,39.5],[-83.5,37.5],[-87.0,36.5],[-88.5,36.0],[-89.0,34.0],[-91.0,33.0],[-91.5,30.5]])
    f("ZOB","Cleveland",[[-77.0,41.5],[-77.0,43.0],[-79.5,43.5],[-82.5,43.5],[-84.0,42.0],[-84.5,41.5],[-84.0,39.0],[-82.5,37.5],[-80.5,39.5],[-77.0,41.5]])
    f("ZID","Indianapolis",[[-84.5,41.5],[-84.0,42.0],[-87.5,41.5],[-87.0,36.5],[-83.5,37.5],[-80.5,39.5],[-82.5,37.5],[-84.0,39.0],[-84.5,41.5]])
    f("ZAU","Chicago",[[-87.5,41.5],[-84.0,42.0],[-82.5,43.5],[-82.5,46.0],[-84.0,46.5],[-87.0,48.0],[-90.0,47.5],[-93.0,46.0],[-93.0,43.5],[-90.0,43.0],[-88.0,43.0],[-87.5,41.5]])
    f("ZMP","Minneapolis",[[-93.0,43.5],[-93.0,46.0],[-90.0,47.5],[-87.0,48.0],[-86.0,49.0],[-95.5,49.0],[-97.5,49.0],[-97.5,45.5],[-96.5,43.5],[-93.0,43.5]])
    f("ZKC","Kansas City",[[-96.5,43.5],[-97.5,45.5],[-100.5,45.5],[-104.0,43.0],[-104.0,40.0],[-102.0,40.0],[-99.5,38.5],[-96.0,37.0],[-93.5,37.0],[-93.0,39.5],[-93.5,43.0],[-96.5,43.5]])
    f("ZME","Memphis",[[-93.5,37.0],[-96.0,37.0],[-96.0,34.5],[-94.5,33.0],[-93.5,29.5],[-91.5,30.5],[-91.0,33.0],[-89.0,34.0],[-88.5,36.0],[-87.0,36.5],[-93.5,37.0]])
    f("ZHU","Houston",[[-97.0,26.0],[-93.5,26.0],[-93.5,29.5],[-94.5,33.0],[-96.0,34.5],[-100.0,33.5],[-100.0,29.0],[-97.0,26.0]])
    f("ZFW","Fort Worth",[[-100.0,33.5],[-96.0,34.5],[-96.0,37.0],[-99.5,38.5],[-102.0,38.5],[-103.0,37.0],[-103.0,34.0],[-100.0,33.5]])
    f("ZDV","Denver",[[-102.0,38.5],[-102.0,40.0],[-104.0,40.0],[-104.0,43.0],[-107.0,43.5],[-109.0,41.0],[-109.0,37.0],[-105.0,37.0],[-103.0,37.0],[-103.0,38.5],[-102.0,38.5]])
    f("ZAB","Albuquerque",[[-109.0,37.0],[-109.0,41.0],[-111.5,41.0],[-114.0,37.5],[-114.0,35.0],[-114.5,32.5],[-111.0,31.5],[-108.0,31.5],[-106.5,31.5],[-104.0,29.5],[-103.5,33.5],[-105.0,37.0],[-109.0,37.0]])
    f("ZLA","Los Angeles",[[-114.0,35.0],[-114.0,37.5],[-120.5,38.5],[-121.5,36.5],[-120.0,34.5],[-118.0,33.0],[-117.0,32.5],[-114.5,32.5],[-114.0,35.0]])
    f("ZOA","Oakland",[[-120.5,38.5],[-114.0,37.5],[-111.5,41.0],[-114.5,42.0],[-117.5,42.0],[-120.5,42.0],[-124.0,40.0],[-122.5,37.0],[-120.5,38.5]])
    f("ZLC","Salt Lake City",[[-111.5,41.0],[-109.0,41.0],[-107.0,43.5],[-111.0,45.0],[-114.0,44.0],[-117.0,44.0],[-117.5,42.0],[-114.5,42.0],[-111.5,41.0]])
    f("ZSE","Seattle",[[-117.0,44.0],[-114.0,44.0],[-111.0,45.0],[-111.0,49.0],[-118.0,49.0],[-124.5,49.0],[-125.0,48.0],[-124.5,43.0],[-120.5,42.0],[-117.5,42.0],[-117.0,44.0]])
    return {"type":"FeatureCollection","features":F}
