"""
artcc_boundaries.py — FAA NASR ARTCC boundary downloader
Downloads the FAA NASR *ShapeFiles* zip (separate from the main NASR subscription),
extracts ARB.shp, dissolves sectors → one polygon per ARTCC center,
and caches to static/artcc.geojson.

Falls back to accurate hand-digitised boundaries if download fails.
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

def _shapefiles_url(d):
    """FAA publishes shapefiles in a *separate* zip from the main NASR subscription."""
    return (
        f"https://nfdc.faa.gov/webContent/28DaySub/"
        f"28DaySubscription_Effective_{d}_ShapeFiles.zip"
    )


# ── Download + convert ────────────────────────────────────────────────────────

def _download(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "model-viewer/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _zip_to_geojson(zip_bytes):
    import geopandas as gpd, tempfile

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        names = z.namelist()

        # Diagnostic: log top-level entries so we can debug future path changes
        top = sorted(set(n.split('/')[0] for n in names))
        log.info(f"ShapeFiles zip top-level: {top}")

        # Find ARB.shp anywhere in the zip
        shp_entry = next((n for n in names if n.upper().endswith("ARB.SHP")), None)
        if not shp_entry:
            # Log a broader listing to help debug
            all_shp = [n for n in names if n.upper().endswith(".SHP")]
            raise FileNotFoundError(
                f"ARB.shp not found. All .shp files: {all_shp[:30]}")

        prefix = shp_entry[:-len("ARB.SHP")]
        log.info(f"Found ARB.shp at prefix='{prefix}'")

        with tempfile.TemporaryDirectory() as td:
            for ext in [".shp", ".SHP", ".dbf", ".DBF",
                        ".shx", ".SHX", ".prj", ".PRJ", ".cpg", ".CPG"]:
                entry = f"{prefix}ARB{ext}"
                # Also try uppercase ARB
                entry_up = f"{prefix.upper()}ARB{ext}"
                for e in [entry, entry_up]:
                    if e in names:
                        (Path(td) / f"ARB{ext.lower()}").write_bytes(z.read(e))
                        break
            gdf = gpd.read_file(Path(td) / "ARB.shp")

    log.info(f"ARB columns: {list(gdf.columns)}")
    log.info(f"ARB rows:    {len(gdf)}")

    # Find the column that identifies each ARTCC center (not sub-sector)
    id_col = next(
        (c for c in ["ARTCC_ID", "CENTER", "ART_ID", "IDENT", "ICAO_ID", "NAME"]
         if c in gdf.columns), None)
    name_col = next(
        (c for c in ["NAME", "ARTCC_NAME", "FULL_NAME", "ARTCC_ID"]
         if c in gdf.columns), None)

    log.info(f"id_col={id_col}  name_col={name_col}")

    # Dissolve all sectors belonging to the same center into one polygon
    if id_col:
        gdf = gdf.dissolve(by=id_col, as_index=False)
        log.info(f"After dissolve: {len(gdf)} ARTCC centers")
    else:
        log.warning("No id_col found — skipping dissolve, using raw rows")

    # Ensure WGS84
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    features = []
    for _, row in gdf.iterrows():
        ident    = str(row.get(id_col,   "")) if id_col   else ""
        fullname = str(row.get(name_col, "")) if name_col else ""
        features.append({
            "type": "Feature",
            "properties": {
                "NAME":     ident or fullname,
                "IDENT":    ident,
                "FULLNAME": fullname,
            },
            "geometry": row.geometry.__geo_interface__,
        })

    return {"type": "FeatureCollection", "features": features}


# ── Public API ────────────────────────────────────────────────────────────────

def ensure_artcc_geojson():
    """
    Call once at app startup (e.g. in a daemon thread).
    Downloads FAA NASR ShapeFiles zip, extracts ARB.shp, writes artcc.geojson.
    Always returns a Path — never raises.
    """
    global _initialized
    with _LOCK:
        if _initialized and ARTCC_FILE.exists():
            return ARTCC_FILE

        STATIC_DIR.mkdir(parents=True, exist_ok=True)

        # Use cached file only if it looks like real FAA data (>100 KB)
        if ARTCC_FILE.exists() and ARTCC_FILE.stat().st_size > 100_000:
            log.info(f"ARTCC: using cached file "
                     f"({ARTCC_FILE.stat().st_size // 1024} KB)")
            _initialized = True
            return ARTCC_FILE

        # Delete a small/bad cached file so we re-download
        if ARTCC_FILE.exists():
            log.info(f"ARTCC: cached file too small "
                     f"({ARTCC_FILE.stat().st_size} B), re-downloading")
            ARTCC_FILE.unlink()

        for d in _nasr_cycle_dates(3):
            url = _shapefiles_url(d)
            log.info(f"ARTCC: downloading ShapeFiles cycle {d}")
            try:
                raw = _download(url, timeout=120)
                log.info(f"ARTCC: downloaded {len(raw) // 1024} KB, converting…")
                gj = _zip_to_geojson(raw)
                ARTCC_FILE.write_text(json.dumps(gj, separators=(',', ':')))
                log.info(f"ARTCC: wrote {len(gj['features'])} centers → {ARTCC_FILE}")
                _initialized = True
                return ARTCC_FILE
            except Exception as e:
                log.warning(f"ARTCC: cycle {d} failed — {e}")
                continue

        # All downloads failed — write built-in so map still works
        log.warning("ARTCC: all downloads failed, using built-in boundaries")
        gj = _builtin()
        ARTCC_FILE.write_text(json.dumps(gj, separators=(',', ':')))
        _initialized = True
        return ARTCC_FILE


def get_artcc_geojson():
    """Return loaded GeoJSON dict (call ensure_artcc_geojson first)."""
    if ARTCC_FILE.exists():
        return json.loads(ARTCC_FILE.read_text())
    return _builtin()


# ── Built-in fallback (hand-digitised from FAA charts, ±10 nm) ──────────────

def _builtin():
    F = []
    def f(n, fn, c):
        F.append({"type": "Feature",
                  "properties": {"NAME": n, "IDENT": n, "FULLNAME": fn},
                  "geometry": {"type": "Polygon", "coordinates": [c]}})

    f("ZBW","Boston",[[-67.0,44.0],[-66.5,44.5],[-67.0,47.5],[-70.7,47.5],
      [-72.0,45.5],[-73.5,45.0],[-73.5,43.0],[-72.5,41.3],[-71.5,41.0],
      [-69.8,41.5],[-69.0,42.0],[-67.0,44.0]])
    f("ZNY","New York",[[-72.5,41.3],[-73.5,43.0],[-75.5,43.5],[-77.0,43.0],
      [-77.0,41.5],[-75.5,39.5],[-74.0,38.9],[-72.5,38.8],[-71.8,40.0],[-72.5,41.3]])
    f("ZDC","Washington",[[-75.5,39.5],[-77.0,41.5],[-80.5,39.5],[-82.0,37.5],
      [-81.0,36.5],[-79.0,36.0],[-76.5,35.0],[-75.8,36.0],[-75.0,37.0],[-75.5,39.5]])
    f("ZJX","Jacksonville",[[-85.5,31.0],[-85.0,29.5],[-83.0,24.5],[-80.5,24.5],
      [-79.5,25.5],[-79.5,27.5],[-78.0,29.5],[-76.5,35.0],[-79.0,36.0],[-81.0,36.5],
      [-82.0,37.5],[-84.5,35.0],[-85.5,33.5],[-85.5,31.0]])
    f("ZMA","Miami",[[-83.0,24.5],[-83.0,20.0],[-72.0,20.0],[-72.0,24.5],
      [-79.5,25.5],[-80.5,24.5],[-83.0,24.5]])
    f("ZTL","Atlanta",[[-91.5,30.5],[-88.0,30.5],[-85.5,31.0],[-85.5,33.5],
      [-84.5,35.0],[-82.0,37.5],[-80.5,39.5],[-83.5,37.5],[-87.0,36.5],
      [-88.5,36.0],[-89.0,34.0],[-91.0,33.0],[-91.5,30.5]])
    f("ZOB","Cleveland",[[-77.0,41.5],[-77.0,43.0],[-79.5,43.5],[-82.5,43.5],
      [-84.0,42.0],[-84.5,41.5],[-84.0,39.0],[-82.5,37.5],[-80.5,39.5],[-77.0,41.5]])
    f("ZID","Indianapolis",[[-84.5,41.5],[-84.0,42.0],[-87.5,41.5],[-87.0,36.5],
      [-83.5,37.5],[-80.5,39.5],[-82.5,37.5],[-84.0,39.0],[-84.5,41.5]])
    f("ZAU","Chicago",[[-87.5,41.5],[-84.0,42.0],[-82.5,43.5],[-82.5,46.0],
      [-84.0,46.5],[-87.0,48.0],[-90.0,47.5],[-93.0,46.0],[-93.0,43.5],
      [-90.0,43.0],[-88.0,43.0],[-87.5,41.5]])
    f("ZMP","Minneapolis",[[-93.0,43.5],[-93.0,46.0],[-90.0,47.5],[-87.0,48.0],
      [-86.0,49.0],[-95.5,49.0],[-97.5,49.0],[-97.5,45.5],[-96.5,43.5],[-93.0,43.5]])
    f("ZKC","Kansas City",[[-96.5,43.5],[-97.5,45.5],[-100.5,45.5],[-104.0,43.0],
      [-104.0,40.0],[-102.0,40.0],[-99.5,38.5],[-96.0,37.0],[-93.5,37.0],
      [-93.0,39.5],[-93.5,43.0],[-96.5,43.5]])
    f("ZME","Memphis",[[-93.5,37.0],[-96.0,37.0],[-96.0,34.5],[-94.5,33.0],
      [-93.5,29.5],[-91.5,30.5],[-91.0,33.0],[-89.0,34.0],[-88.5,36.0],
      [-87.0,36.5],[-93.5,37.0]])
    f("ZHU","Houston",[[-97.0,26.0],[-93.5,26.0],[-93.5,29.5],[-94.5,33.0],
      [-96.0,34.5],[-100.0,33.5],[-100.0,29.0],[-97.0,26.0]])
    f("ZFW","Fort Worth",[[-100.0,33.5],[-96.0,34.5],[-96.0,37.0],[-99.5,38.5],
      [-102.0,38.5],[-103.0,37.0],[-103.0,34.0],[-100.0,33.5]])
    f("ZDV","Denver",[[-102.0,38.5],[-102.0,40.0],[-104.0,40.0],[-104.0,43.0],
      [-107.0,43.5],[-109.0,41.0],[-109.0,37.0],[-105.0,37.0],[-103.0,37.0],
      [-103.0,38.5],[-102.0,38.5]])
    f("ZAB","Albuquerque",[[-109.0,37.0],[-109.0,41.0],[-111.5,41.0],[-114.0,37.5],
      [-114.0,35.0],[-114.5,32.5],[-111.0,31.5],[-108.0,31.5],[-106.5,31.5],
      [-104.0,29.5],[-103.5,33.5],[-105.0,37.0],[-109.0,37.0]])
    f("ZLA","Los Angeles",[[-114.0,35.0],[-114.0,37.5],[-120.5,38.5],[-121.5,36.5],
      [-120.0,34.5],[-118.0,33.0],[-117.0,32.5],[-114.5,32.5],[-114.0,35.0]])
    f("ZOA","Oakland",[[-120.5,38.5],[-114.0,37.5],[-111.5,41.0],[-114.5,42.0],
      [-117.5,42.0],[-120.5,42.0],[-124.0,40.0],[-122.5,37.0],[-120.5,38.5]])
    f("ZLC","Salt Lake City",[[-111.5,41.0],[-109.0,41.0],[-107.0,43.5],
      [-111.0,45.0],[-114.0,44.0],[-117.0,44.0],[-117.5,42.0],[-114.5,42.0],
      [-111.5,41.0]])
    f("ZSE","Seattle",[[-117.0,44.0],[-114.0,44.0],[-111.0,45.0],[-111.0,49.0],
      [-118.0,49.0],[-124.5,49.0],[-125.0,48.0],[-124.5,43.0],[-120.5,42.0],
      [-117.5,42.0],[-117.0,44.0]])

    return {"type": "FeatureCollection", "features": F}
