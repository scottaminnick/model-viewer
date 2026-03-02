"""
app.py  –  Model Viewer
HRRR Colorado forecast visualization.
Root redirects to the interactive map at /map/hrrr.
"""

import os
import traceback
from flask import Flask, jsonify, redirect, render_template_string, Response, request

from winds         import get_hrrr_gusts_cached, get_cycle_status_cached
from froude        import get_froude_cached
from icing         import get_icing_cached
from winds_surface import get_surface_wind_cached
from virga         import get_virga_cached
from llti          import get_llti_cached, get_llti_points_cached
from prefetch      import start_prefetch_thread, get_all_status
from rap_conus     import (get_rap_conus_cached, get_rap_cycle_status_cached,
                       get_rap_conus_image_cached)
from artcc_boundaries import get_artcc_geojson


app = Flask(__name__)

# Start background pre-fetcher (warms F01-F12 cache for all products)
start_prefetch_thread()


# ─────────────────────────────────────────────────────────────────────────────
#  Root
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return redirect("/map/hrrr")


# ─────────────────────────────────────────────────────────────────────────────
#  Map pages
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/map/hrrr")
def map_hrrr():
    return render_template_string(HRRR_MAP_TEMPLATE)

@app.get("/map/winds")
def map_winds():
    return redirect("/map/hrrr?product=winds")

@app.get("/map/froude")
def map_froude():
    return redirect("/map/hrrr?product=froude")

@app.get("/map/virga")
def map_virga():
    return redirect("/map/hrrr?product=virga")

@app.get("/map/icing")
def map_icing():
    return redirect("/map/hrrr?product=icing")

@app.get("/map/surface")
def map_surface():
    return redirect("/map/hrrr?product=surface_wind")

@app.get("/map/llti")
def map_llti():
    return redirect("/map/hrrr?product=llti")


# ─────────────────────────────────────────────────────────────────────────────
#  Ops
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return jsonify(status="ok")

@app.get("/debug/routes")
def debug_routes():
    return jsonify(sorted(str(r) for r in app.url_map.iter_rules()))

@app.get("/api/cache/status")
def api_cache_status():
    return jsonify(get_all_status())


# ─────────────────────────────────────────────────────────────────────────────
#  HRRR cycle status  (drives the hour-button UI)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/winds/status")
def api_winds_status():
    ttl = int(os.environ.get("STATUS_TTL", "300"))
    return jsonify(get_cycle_status_cached(ttl_seconds=ttl))


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
def _not_ready(msg: str) -> bool:
    return any(k in msg.lower() for k in [
        "did not find", "not found", "no such file", "404", "unavailable",
        "nomads", "full file", "byte-range", "grib_lock",
    ])

def _resolve_cycle(cycle_utc: str | None) -> str:
    if cycle_utc:
        return cycle_utc
    status = get_cycle_status_cached(ttl_seconds=300)
    return status["cycles"][0]["cycle_utc"]


# ─────────────────────────────────────────────────────────────────────────────
#  Product data endpoints
#  All follow the same pattern:
#    GET /api/<product>/colorado?fxx=<int>&cycle_utc=<str>
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/winds/colorado")
def api_winds_colorado():
    fxx       = int(request.args.get("fxx", 1))
    cycle_utc = _resolve_cycle(request.args.get("cycle_utc"))
    ttl       = int(request.args.get("ttl", os.environ.get("WINDS_TTL", "600")))
    try:
        return jsonify(get_hrrr_gusts_cached(cycle_utc=cycle_utc, fxx=fxx, ttl_seconds=ttl))
    except Exception as e:
        if _not_ready(str(e)):
            return jsonify({"error": "not_available", "fxx": fxx,
                            "cycle_utc": cycle_utc,
                            "message": f"F{fxx:02d} not yet available."}), 404
        raise


@app.get("/api/froude/colorado")
def api_froude_colorado():
    fxx       = int(request.args.get("fxx", 1))
    cycle_utc = _resolve_cycle(request.args.get("cycle_utc"))
    ttl       = int(request.args.get("ttl", os.environ.get("FROUDE_TTL", "600")))
    try:
        return jsonify(get_froude_cached(cycle_utc=cycle_utc, fxx=fxx, ttl_seconds=ttl))
    except Exception as e:
        if _not_ready(str(e)):
            return jsonify({"error": "not_available", "fxx": fxx,
                            "cycle_utc": cycle_utc,
                            "message": f"F{fxx:02d} not yet available."}), 404
        raise


@app.get("/api/virga/colorado")
def api_virga_colorado():
    fxx       = int(request.args.get("fxx", 1))
    cycle_utc = _resolve_cycle(request.args.get("cycle_utc"))
    ttl       = int(request.args.get("ttl", os.environ.get("VIRGA_TTL", "600")))
    try:
        return jsonify(get_virga_cached(cycle_utc=cycle_utc, fxx=fxx, ttl_seconds=ttl))
    except Exception as e:
        if _not_ready(str(e)):
            return jsonify({"error": "not_available", "fxx": fxx,
                            "cycle_utc": cycle_utc,
                            "message": f"F{fxx:02d} not yet available."}), 404
        raise


@app.get("/api/icing/colorado")
def api_icing_colorado():
    fxx       = int(request.args.get("fxx", 1))
    cycle_utc = _resolve_cycle(request.args.get("cycle_utc"))
    ttl       = int(request.args.get("ttl", os.environ.get("ICING_TTL", "600")))
    try:
        return jsonify(get_icing_cached(cycle_utc=cycle_utc, fxx=fxx, ttl_seconds=ttl))
    except Exception as e:
        if _not_ready(str(e)):
            return jsonify({"error": "not_available", "fxx": fxx,
                            "cycle_utc": cycle_utc,
                            "message": f"F{fxx:02d} not yet available."}), 404
        raise


@app.get("/api/winds/surface")
def api_winds_surface():
    fxx       = int(request.args.get("fxx", 1))
    cycle_utc = _resolve_cycle(request.args.get("cycle_utc"))
    ttl       = int(request.args.get("ttl", os.environ.get("WIND_SURF_TTL", "600")))
    try:
        return jsonify(get_surface_wind_cached(cycle_utc=cycle_utc, fxx=fxx, ttl_seconds=ttl))
    except Exception as e:
        if _not_ready(str(e)):
            return jsonify({"error": "not_available", "fxx": fxx,
                            "cycle_utc": cycle_utc,
                            "message": f"F{fxx:02d} not yet available."}), 404
        raise


@app.get("/api/llti/colorado")
def api_llti_colorado():
    fxx       = int(request.args.get("fxx", 1))
    cycle_utc = _resolve_cycle(request.args.get("cycle_utc"))
    ttl       = int(request.args.get("ttl", os.environ.get("LLTI_TTL", "600")))
    try:
        return jsonify(get_llti_points_cached(cycle_utc=cycle_utc, fxx=fxx, ttl_seconds=ttl))
    except Exception as e:
        if _not_ready(str(e)):
            return jsonify({"error": "not_available", "fxx": fxx,
                            "cycle_utc": cycle_utc,
                            "message": f"F{fxx:02d} not yet available."}), 404
        raise


@app.get("/api/llti/image")
def api_llti_image():
    ttl = int(os.environ.get("LLTI_TTL", "600"))
    try:
        png_bytes, _ = get_llti_cached(ttl_seconds=ttl)
        return Response(png_bytes, mimetype="image/png")
    except Exception:
        return Response(traceback.format_exc(), mimetype="text/plain", status=500)


@app.get("/api/llti/meta")
def api_llti_meta():
    ttl = int(os.environ.get("LLTI_TTL", "600"))
    try:
        _, meta = get_llti_cached(ttl_seconds=ttl)
        return jsonify(meta)
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500

# ── RAP CONUS map page ────────────────────────────────────────────────────────
@app.get("/map/conus")
def map_conus():
    return render_template_string(CONUS_MAP_TEMPLATE)


# ── RAP cycle status (drives hour buttons in CONUS map) ───────────────────────
@app.get("/api/rap/status")
def api_rap_status():
    ttl = int(os.environ.get("RAP_STATUS_TTL", "300"))
    return jsonify(get_rap_cycle_status_cached(ttl_seconds=ttl))

@app.get("/api/artcc/boundaries")
def api_artcc_boundaries():
    """Serve ARTCC boundary GeoJSON — tries FAA live endpoint, falls back to built-in."""
    data = get_artcc_geojson(ttl=86400)   # cache for 24 h
    resp = jsonify(data)
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


# ── RAP CONUS gust data ───────────────────────────────────────────────────────
@app.get("/api/rap/conus")
def api_rap_conus():
    fxx       = int(request.args.get("fxx", 1))
    ttl       = int(request.args.get("ttl", os.environ.get("RAP_TTL", "600")))

    # Resolve cycle: use provided or latest available
    cycle_utc = request.args.get("cycle_utc")
    if not cycle_utc:
        status = get_rap_cycle_status_cached(ttl_seconds=300)
        cycle_utc = status["cycles"][0]["cycle_utc"]

    try:
        return jsonify(get_rap_conus_cached(cycle_utc=cycle_utc, fxx=fxx, ttl_seconds=ttl))
    except Exception as e:
        if _not_ready(str(e)):
            return jsonify({
                "error":     "not_available",
                "fxx":       fxx,
                "cycle_utc": cycle_utc,
                "message":   f"F{fxx:02d} not yet available on AWS.",
            }), 404
        raise

@app.get("/api/rap/conus/image")
def api_rap_conus_image():
    fxx       = int(request.args.get("fxx", 1))
    ttl       = int(request.args.get("ttl", os.environ.get("RAP_TTL", "600")))

    cycle_utc = request.args.get("cycle_utc")
    if not cycle_utc:
        status    = get_rap_cycle_status_cached(ttl_seconds=300)
        cycle_utc = status["cycles"][0]["cycle_utc"]

    try:
        png_bytes, meta = get_rap_conus_image_cached(
            cycle_utc=cycle_utc, fxx=fxx, ttl_seconds=ttl)
        resp = Response(png_bytes, mimetype="image/png")
        resp.headers["Cache-Control"] = "public, max-age=600"
        resp.headers["X-Valid-UTC"]   = meta.get("valid_utc", "")
        return resp
    except Exception as e:
        if _not_ready(str(e)):
            return jsonify({"error": "not_available", "fxx": fxx,
                            "message": f"F{fxx:02d} not yet available."}), 404
        raise

# ─────────────────────────────────────────────────────────────────────────────
#  Debug helpers
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/debug/prs_fields")
def debug_prs_fields():
    """Dump pressure-level fields in the latest HRRR prs product."""
    import pygrib
    from winds import _find_latest_hrrr_cycle, HERBIE_DIR
    from herbie import Herbie
    from pathlib import Path
    cycle = _find_latest_hrrr_cycle()
    H = Herbie(cycle, model="hrrr", product="prs", fxx=1,
               save_dir=str(HERBIE_DIR), overwrite=False)
    grib_path = Path(H.download())
    grbs = pygrib.open(str(grib_path))
    fields = [{"name": g.name, "shortName": g.shortName,
               "typeOfLevel": g.typeOfLevel, "level": g.level} for g in grbs]
    grbs.close()
    froude_fields = [f for f in fields
                     if any(k in f["name"].lower() for k in
                            ["wind","temperature","geopotential","height",
                             "u-component","v-component"])
                     and f["typeOfLevel"] == "isobaricInhPa"]
    return jsonify({"cycle": cycle.isoformat(),
                    "pressure_levels_mb": sorted({f["level"] for f in froude_fields}),
                    "froude_relevant_fields": froude_fields})


@app.get("/debug/sfc_fields")
def debug_sfc_fields():
    """Dump surface fields in the latest HRRR sfc product."""
    import pygrib
    from winds import _find_latest_hrrr_cycle, HERBIE_DIR
    from herbie import Herbie
    from pathlib import Path
    cycle = _find_latest_hrrr_cycle()
    H = Herbie(cycle, model="hrrr", product="sfc", fxx=1,
               save_dir=str(HERBIE_DIR), overwrite=False)
    grib_path = Path(H.download())
    grbs = pygrib.open(str(grib_path))
    fields = [{"name": g.name, "shortName": g.shortName,
               "typeOfLevel": g.typeOfLevel, "level": g.level}
              for g in grbs
              if any(k in g.name.lower()
                     for k in ["wind","gust","boundary","temperature",
                                "height","cloud","dewpoint"])]
    grbs.close()
    return jsonify({"cycle": cycle.isoformat(), "fields": fields})


# ─────────────────────────────────────────────────────────────────────────────
#  Global error handler
# ─────────────────────────────────────────────────────────────────────────────
@app.errorhandler(Exception)
def handle_exception(e):
    return Response(traceback.format_exc(), mimetype="text/plain", status=500)


# ─────────────────────────────────────────────────────────────────────────────
#  Map template
# ─────────────────────────────────────────────────────────────────────────────
HRRR_MAP_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Model Viewer — HRRR Colorado</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --border:#30363d;
          --text:#e6edf3; --muted:#8b949e; --ac:#58a6ff; }
  *,*::before,*::after { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:system-ui,sans-serif;
         height:100dvh; display:flex; flex-direction:column; overflow:hidden; }
  #header { background:var(--panel); border-bottom:1px solid var(--border);
    padding:0.45rem 0.75rem; display:flex; align-items:center;
    gap:0.75rem; flex-wrap:wrap; flex-shrink:0; }
  #header .title { font-weight:700; font-size:0.95rem; white-space:nowrap; }
  select,input[type=range] { background:var(--bg); color:var(--text);
    border:1px solid var(--border); border-radius:5px; font-size:0.78rem; }
  select { padding:0.28rem 0.5rem; cursor:pointer; }
  .ctrl-group { display:flex; align-items:center; gap:0.4rem; }
  .ctrl-label { font-size:0.68rem; color:var(--muted); white-space:nowrap; }
  #product-sel { font-weight:600; color:var(--ac); border-color:var(--ac); padding:0.3rem 0.6rem; }
  #hour-bar { display:flex; align-items:center; gap:0.3rem; padding:0.3rem 0.75rem;
    background:var(--panel); border-bottom:1px solid var(--border);
    flex-shrink:0; flex-wrap:wrap; }
  .hbtn { font-size:0.72rem; font-weight:600; padding:0.22rem 0.5rem;
    border-radius:4px; border:1px solid var(--border); background:var(--bg);
    color:var(--muted); cursor:pointer; transition:background 0.15s,color 0.15s;
    position:relative; }
  .hbtn.available { color:var(--text); border-color:#444; }
  .hbtn.active { background:var(--ac); color:#000; border-color:var(--ac); }
  .hbtn.unavail { opacity:0.35; cursor:not-allowed; }
  .dot-badge { position:absolute; top:-3px; right:-3px;
               width:6px; height:6px; border-radius:50%; }
  .dot-green { background:#2ecc71; } .dot-grey { background:#555; }
  #progress-bar { height:3px; background:var(--border); flex:1;
                  border-radius:2px; min-width:60px; }
  #progress-fill { height:100%; background:var(--ac); border-radius:2px;
                   transition:width 0.4s; width:0%; }
  #main { flex:1; display:flex; min-height:0; }
  #map  { flex:1; }
  #sidebar { width:210px; background:var(--panel); border-left:1px solid var(--border);
             display:flex; flex-direction:column; flex-shrink:0; overflow-y:auto; }
  #legend { padding:0.75rem; }
  .leg-title { font-size:0.72rem; font-weight:700; color:var(--muted); margin-bottom:0.5rem; }
  .leg-row { display:flex; align-items:center; gap:0.55rem; margin:0.3rem 0; }
  .leg-swatch { width:22px; height:13px; border-radius:3px; opacity:0.85; flex-shrink:0; }
  .leg-sub { font-size:0.65rem; color:var(--muted); margin-left:auto; }
  #meta { padding:0.5rem 0.75rem; font-size:0.68rem; color:var(--muted);
          border-top:1px solid var(--border); }
  #meta b { color:var(--text); }
  #loading-overlay { position:absolute; inset:0; z-index:2000;
    background:rgba(13,17,23,0.88); display:flex; flex-direction:column;
    align-items:center; justify-content:center; gap:1rem; transition:opacity 0.3s; }
  #loading-overlay.hidden { opacity:0; pointer-events:none; }
  .spinner { width:42px; height:42px; border:3px solid var(--border);
             border-top-color:var(--ac); border-radius:50%;
             animation:spin 0.8s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  #load-msg { font-size:0.8rem; color:var(--muted); text-align:center; max-width:240px; }
  .apt-label { background:none!important; border:none!important; box-shadow:none!important;
    font-size:0.65rem; font-weight:700; color:#58a6ff;
    text-shadow:0 0 3px #0d1117; padding:0!important; }
  .city-label { background:none!important; border:none!important; box-shadow:none!important;
    font-size:0.62rem; color:#8b949e; text-shadow:0 0 3px #0d1117; padding:0!important; }
  .city-label-major { background:none!important; border:none!important;
    box-shadow:none!important; font-size:0.72rem; font-weight:600; color:#e6edf3;
    text-shadow:0 0 4px #0d1117; padding:0!important; }
  .leaflet-control-layers { background:var(--panel)!important;
    border:1px solid var(--border)!important; color:var(--text)!important; font-size:0.78rem; }
  .leaflet-control-layers label { color:var(--text)!important; }
  .leaflet-control-layers-overlays { padding:0.2rem 0.4rem; }
  #error-bar { display:none; background:#5a1a1a; color:#f9a8a8;
               padding:0.4rem 0.75rem; font-size:0.78rem;
               border-bottom:1px solid #8b2020; }
</style>
</head>
<body>

<div id="header">
  <span class="title">🏔 HRRR Colorado</span>
  <div class="ctrl-group">
    <span class="ctrl-label">PRODUCT</span>
    <select id="product-sel" onchange="onProductChange()">
      <option value="winds">Wind Gusts</option>
      <option value="froude">Froude Number</option>
      <option value="virga">Virga Potential</option>
      <option value="icing">Icing Threat</option>
      <option value="surface_wind">Surface Flow</option>
      <option value="llti">LLTI</option>
    </select>
  </div>
  <div class="ctrl-group">
    <span class="ctrl-label">CYCLE</span>
    <select id="cycle-sel" onchange="onCycleChange()"><option value="">—</option></select>
  </div>
  <div class="ctrl-group" style="margin-left:auto;">
    <span class="ctrl-label">OPACITY</span>
    <input type="range" id="opacity-slider" min="10" max="100" step="5" value="65"
      style="width:80px;" oninput="updateOpacity(this.value)"/>
    <span id="opacity-val" style="font-size:0.72rem;color:var(--muted);width:28px;">65%</span>
  </div>
</div>

<div id="error-bar"></div>

<div id="hour-bar">
  <span class="ctrl-label">HOUR →</span>
  <div id="progress-bar"><div id="progress-fill"></div></div>
  <span id="cycle-pct" style="font-size:0.68rem;color:var(--muted);white-space:nowrap;"></span>
</div>

<div id="main">
  <div id="map" style="position:relative;">
    <div id="loading-overlay">
      <div class="spinner"></div>
      <div id="load-msg">Loading…</div>
    </div>
  </div>
  <div id="sidebar">
    <div id="legend"></div>
    <div id="meta">
      <div>Valid: <b id="meta-valid">—</b></div>
      <div>Points: <b id="meta-pts">—</b></div>
      <div style="margin-top:0.4rem;font-size:0.63rem;">Click any cell for details</div>
    </div>
  </div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const PRODUCTS = {
  winds: {
    label:'Wind Gusts', endpoint:'/api/winds/colorado',
    loadMsg:'Fetching HRRR sfc…<br><small style="color:var(--muted)">~15 s first load</small>',
    color:function(p){ return p.gust_kt>=50?'#e74c3c':p.gust_kt>=35?'#e67e22':p.gust_kt>=20?'#f1c40f':'#2ecc71'; },
    popup:function(p){ return '<b>'+p.gust_kt.toFixed(0)+' kt gust</b><br>'+p.lat.toFixed(3)+'°N, '+Math.abs(p.lon).toFixed(3)+'°W'; },
    legend:`<div class="leg-title">Wind Gust (kt)</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#2ecc71"></div>&lt; 20 kt</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#f1c40f"></div>20–35 kt</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#e67e22"></div>35–50 kt</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#e74c3c"></div>&ge; 50 kt</div>`
  },
  froude: {
    label:'Froude Number', endpoint:'/api/froude/colorado',
    loadMsg:'Fetching HRRR prs…<br><small style="color:var(--muted)">~60 s first load</small>',
    color:function(p){ return p.cat===3?'#e91e8c':p.cat===2?'#00bcd4':p.cat===4?'#7b1fa2':'#2ecc71'; },
    popup:function(p){
      return '<b>Fr = '+p.fr.toFixed(2)+'</b><br>Wind 700mb: '+p.wind_kt.toFixed(0)+' kt<br>'+
             'N: '+(p.N*1000).toFixed(2)+' &times; 10⁻³ s⁻¹<br>Terrain h: '+p.h_m.toFixed(0)+' m<br>'+
             p.lat.toFixed(3)+'°N, '+Math.abs(p.lon).toFixed(3)+'°W';
    },
    legend:`<div class="leg-title">Froude Number  Fr = U / (N &times; h)</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#2ecc71"></div>Fr &lt; 0.5 &mdash; Splitting <span class="leg-sub">low</span></div>
  <div class="leg-row"><div class="leg-swatch" style="background:#00bcd4"></div>0.5 &le; Fr &lt; 0.8 &mdash; Transitional</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#e91e8c"></div>0.8 &le; Fr &le; 1.5 &mdash; Resonant <span class="leg-sub">HIGH</span></div>
  <div class="leg-row"><div class="leg-swatch" style="background:#7b1fa2"></div>Fr &gt; 1.5 &mdash; Flow over</div>`
  },
  virga: {
    label:'Virga Potential', endpoint:'/api/virga/colorado',
    loadMsg:'Fetching HRRR prs…<br><small style="color:var(--muted)">~90 s first load</small>',
    color:function(p){ return p.cat>=4?'#8e44ad':p.cat>=3?'#e74c3c':p.cat>=2?'#e67e22':p.cat>=1?'#f1c40f':'#2c3e50'; },
    popup:function(p){
      return '<b>Virga: '+p.virga_pct.toFixed(0)+'%</b><br>CB Wind: '+p.cb_wind_kt.toFixed(0)+' kt<br>'+
             'Upper RH: '+p.upper_rh.toFixed(0)+'%<br>'+p.lat.toFixed(3)+'°N, '+Math.abs(p.lon).toFixed(3)+'°W';
    },
    legend:`<div class="leg-title">Virga Potential</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#2c3e50"></div>Negligible</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#f1c40f"></div>20–40% &mdash; Low</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#e67e22"></div>40–60% &mdash; Moderate</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#e74c3c"></div>60–80% &mdash; High</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#8e44ad"></div>&ge; 80% &mdash; Extreme</div>`
  },
  icing: {
    label:'Icing Threat', endpoint:'/api/icing/colorado',
    loadMsg:'Fetching HRRR prs…<br><small style="color:var(--muted)">RH + omega + convergence</small>',
    color:function(p){ return p.cat>=3?'#e74c3c':p.cat>=2?'#e67e22':p.cat>=1?'#f1c40f':'#2c3e50'; },
    popup:function(p){
      var up='';
      if(p.wdir850>=45&&p.wdir850<=135&&p.spd850>=10) up='<br><b style="color:#58a6ff">▲ Front Range upslope</b>';
      if(p.wdir850>=225&&p.wdir850<=315&&p.spd850>=10) up='<br><b style="color:#58a6ff">▲ West slope upslope</b>';
      return '<b>Icing score: '+p.score.toFixed(2)+'</b> (cat '+p.cat+')<br>'+
             'RH 850/700: '+p.rh850.toFixed(0)+'% / '+p.rh700.toFixed(0)+'%<br>'+
             '850mb wind: '+p.spd850.toFixed(0)+' kt @ '+p.wdir850.toFixed(0)+'°'+up+'<br>'+
             p.lat.toFixed(3)+'°N, '+Math.abs(p.lon).toFixed(3)+'°W';
    },
    legend:`<div class="leg-title">Winter Icing Threat Index</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#2c3e50"></div>Negligible <span class="leg-sub">&lt;0.35</span></div>
  <div class="leg-row"><div class="leg-swatch" style="background:#f1c40f"></div>Low <span class="leg-sub">0.35–0.55</span></div>
  <div class="leg-row"><div class="leg-swatch" style="background:#e67e22"></div>Moderate <span class="leg-sub">0.55–0.75</span></div>
  <div class="leg-row"><div class="leg-swatch" style="background:#e74c3c"></div>High <span class="leg-sub">&ge;0.75</span></div>
  <div style="margin-top:0.6rem;font-size:0.63rem;color:var(--muted);">Sat(0.45)·Ascent(0.35)·Conv(0.20)<br>+0.15 Front Range·+0.10 West slope</div>`
  },
  surface_wind: {
    label:'Surface Flow', endpoint:'/api/winds/surface',
    loadMsg:'Fetching HRRR 10m wind…<br><small style="color:var(--muted)">~15 s</small>',
    renderMode:'streamline',
    color:function(p){ return p.cat>=4?'#e74c3c':p.cat>=3?'#e67e22':p.cat>=2?'#f1c40f':p.cat>=1?'#3d8f6e':'#1a3a5c'; },
    popup:function(p){ return '<b>'+p.spd.toFixed(0)+' kt</b> from '+p.wdir.toFixed(0)+'°<br>'+p.lat.toFixed(3)+'°N, '+Math.abs(p.lon).toFixed(3)+'°W'; },
    legend:`<div class="leg-title">10m Wind Speed</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#1a3a5c"></div>&lt; 8 kt</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#3d8f6e"></div>8–15 kt</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#f1c40f"></div>15–25 kt</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#e67e22"></div>25–40 kt</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#e74c3c"></div>&ge; 40 kt</div>
  <div style="margin-top:0.6rem;font-size:0.63rem;color:var(--muted);">White streamlines show flow direction.</div>`
  },
  llti: {
    label:'LLTI', endpoint:'/api/llti/colorado',
    loadMsg:'Fetching HRRR LLTI…<br><small style="color:var(--muted)">~90 s first load</small>',
    color:function(p){ return p.cat>=3?'#e74c3c':p.cat>=2?'#FF8C00':p.cat>=1?'#FFD700':'#006400'; },
    popup:function(p){
      return '<b>LLTI: '+p.llti.toFixed(0)+'</b> (cat '+p.cat+')<br>'+
             'Mix Hgt: '+p.mix_ft.toFixed(0)+' ft<br>Transport Wind: '+p.trspd_kt.toFixed(1)+' kt<br>'+
             'Sky: '+p.sky_pct.toFixed(0)+'%<br>Dewpoint Dep: '+p.dd_f.toFixed(1)+'°F<br>'+
             p.lat.toFixed(3)+'°N, '+Math.abs(p.lon).toFixed(3)+'°W';
    },
    legend:`<div class="leg-title">Low-Level Turbulence Index</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#006400"></div>&lt; 25 &mdash; Negligible</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#FFD700"></div>25–50 &mdash; Low</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#FF8C00"></div>50–75 &mdash; Moderate</div>
  <div class="leg-row"><div class="leg-swatch" style="background:#e74c3c"></div>&ge; 75 &mdash; High</div>
  <div style="margin-top:0.6rem;font-size:0.63rem;color:var(--muted);">MixHgt(0.25)·TransWind(0.45)<br>Sky(0.15)·DewDep(0.15)<br>HPBL-coupled 10m+950–700mb</div>`
  }
};

var currentProduct='winds', currentCycle=null, currentFxx=1;
var currentOpacity=0.65, cycleStatus={}, dataLayer=null;

var map=L.map('map',{center:[39.0,-105.5],zoom:7});
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}',
  {attribution:'Tiles &copy; Esri',maxZoom:13}).addTo(map);
var roadsLayer=L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}',
  {attribution:'',maxZoom:13,opacity:0.55});

var CO_AIRPORTS=[
  ["KDEN","Denver Intl",39.8561,-104.6737,"com"],["KCOS","Colorado Springs",38.8059,-104.7008,"com"],
  ["KGJT","Grand Junction",39.1224,-108.5268,"com"],["KDRO","Durango La Plata",37.1515,-107.7538,"com"],
  ["KPUB","Pueblo Memorial",38.2891,-104.4966,"com"],["KASE","Aspen/Pitkin",39.2232,-106.8687,"com"],
  ["KEGE","Eagle County",39.6426,-106.9177,"com"],["KHDN","Yampa Valley",40.4812,-107.2218,"com"],
  ["KGUC","Gunnison-Crested Butte",38.5339,-106.9330,"com"],["KMTJ","Montrose Regional",38.5098,-107.8938,"com"],
  ["KALS","San Luis Valley",37.4349,-105.8666,"com"],["KTEX","Telluride Regional",37.9538,-107.9088,"com"],
  ["KFNL","Northern CO Regional",40.4518,-105.0110,"com"],["KAPA","Centennial",39.5701,-104.8490,"ga"],
  ["KBJC","Rocky Mtn Metro",39.9088,-105.1172,"ga"],["KBDU","Boulder Municipal",40.0394,-105.2257,"ga"],
  ["KANK","Harriet Alexander",38.5398,-105.9952,"ga"],["KAEJ","Central CO Regional",38.8440,-106.1188,"ga"],
  ["KLXV","Lake County",39.2238,-106.3177,"ga"],["KRIL","Garfield Co",39.5263,-107.7266,"ga"],
  ["KCAG","Craig-Moffat",40.4952,-107.5225,"ga"],["KTAD","Perry Stokes",37.2594,-104.3412,"ga"],
  ["KLIC","Limon Municipal",39.2748,-103.6659,"ga"],["KLAA","Lamar Municipal",38.0697,-102.6886,"ga"],
];
function buildAirportLayer(){
  var markers=[];
  CO_AIRPORTS.forEach(function(a){
    var isCom=a[4]==="com";
    var m=L.circleMarker([a[2],a[3]],{radius:isCom?7:5,color:isCom?"#58a6ff":"#8b949e",
      fillColor:isCom?"#1f6feb":"#30363d",fillOpacity:0.85,weight:1.5});
    m.bindPopup('<b>'+a[0]+'</b><br>'+a[1],{maxWidth:180});
    markers.push(m);
  });
  var layer=L.layerGroup(markers);
  map.on("zoomend",function(){
    if(!map.hasLayer(layer)) return;
    markers.forEach(function(m){
      if(map.getZoom()>=8){
        if(!m.getTooltip()){
          var icao=m.getPopup().getContent().replace(/<b>(.*?)<[/]b>.*/,'$1');
          m.bindTooltip(icao,{permanent:true,direction:"right",className:"apt-label",offset:[6,0]}).openTooltip();
        }
      } else { if(m.getTooltip()) m.unbindTooltip(); }
    });
  });
  return layer;
}
var CO_CITIES=[
  ["Denver",39.7392,-104.9903,true],["Colorado Springs",38.8339,-104.8214,true],
  ["Grand Junction",39.0639,-108.5506,true],["Pueblo",38.2544,-104.6091,false],
  ["Fort Collins",40.5853,-105.0844,false],["Boulder",40.0150,-105.2705,false],
  ["Steamboat Springs",40.4850,-106.8317,false],["Glenwood Springs",39.5505,-107.3248,false],
  ["Aspen",39.1911,-106.8175,false],["Telluride",37.9375,-107.8123,false],
  ["Montrose",38.4783,-107.8762,false],["Alamosa",37.4695,-105.8700,false],
  ["Durango",37.2753,-107.8801,false],["Salida",38.5347,-106.0000,false],
  ["Leadville",39.2503,-106.2925,false],["Craig",40.5153,-107.5464,false],
];
function buildCityLayer(){
  var markers=[];
  CO_CITIES.forEach(function(c){
    var m=L.circleMarker([c[1],c[2]],{radius:c[3]?5:3,color:"#e6edf3",fillColor:"#e6edf3",
      fillOpacity:c[3]?0.9:0.6,weight:1});
    m.bindTooltip(c[0],{permanent:true,direction:"right",
      className:c[3]?"city-label-major":"city-label",offset:[5,0]});
    markers.push(m);
  });
  return L.layerGroup(markers);
}
var airportLayer=buildAirportLayer(), cityLayer=buildCityLayer();
L.control.layers(null,{"✈ Airports":airportLayer,"● Cities":cityLayer,"≡ Roads":roadsLayer},
  {collapsed:false,position:"topright"}).addTo(map);

function onProductChange(){ currentProduct=document.getElementById('product-sel').value; updateLegend(); if(currentCycle) loadData(); }
function updateLegend(){ document.getElementById('legend').innerHTML=PRODUCTS[currentProduct].legend; }
function updateOpacity(val){ currentOpacity=val/100; document.getElementById('opacity-val').textContent=val+'%';
  if(dataLayer) dataLayer.eachLayer(function(l){l.setStyle({fillOpacity:currentOpacity});}); }

async function fetchStatus(){
  try{
    var s=await(await fetch('/api/winds/status')).json();
    cycleStatus={};
    (s.cycles||[]).forEach(function(c){cycleStatus[c.cycle_utc]=c;});
    var sel=document.getElementById('cycle-sel'), prev=sel.value;
    sel.innerHTML='';
    Object.keys(cycleStatus).sort().reverse().forEach(function(c){
      var opt=document.createElement('option'); opt.value=c;
      opt.textContent=new Date(c).toUTCString().slice(5,22)+'Z'; sel.appendChild(opt);
    });
    if(prev&&cycleStatus[prev]) sel.value=prev;
    else if(!currentCycle&&sel.options.length){sel.value=sel.options[0].value; currentCycle=sel.value;}
    buildHourButtons();
    var cs=cycleStatus[currentCycle];
    if(cs){document.getElementById('progress-fill').style.width=cs.pct_complete+'%';
            document.getElementById('cycle-pct').textContent=cs.pct_complete+'% ready';}
  }catch(e){console.warn('status fetch failed',e);}
}
function onCycleChange(){ currentCycle=document.getElementById('cycle-sel').value; buildHourButtons(); loadData(); }
function buildHourButtons(){
  document.querySelectorAll('.hbtn').forEach(function(b){b.remove();});
  var cs=cycleStatus[currentCycle], avail=cs?cs.available_hours:[];
  var cache=cs?(cs.cached_hours||{}):{};
  var bar=document.getElementById('hour-bar'), prog=document.getElementById('progress-bar');
  for(var fxx=1;fxx<=12;fxx++){(function(f){
    var btn=document.createElement('button');
    btn.className='hbtn'; btn.textContent='F'+String(f).padStart(2,'0'); btn.dataset.fxx=f;
    var dot=document.createElement('span'); dot.className='dot-badge';
    dot.classList.add((cache[currentProduct]&&cache[currentProduct].includes(f))?'dot-green':'dot-grey');
    btn.appendChild(dot);
    if(avail.includes(f)){btn.classList.add('available'); btn.onclick=function(){selectHour(f);};}
    else{btn.classList.add('unavail'); btn.disabled=true;}
    if(f===currentFxx) btn.classList.add('active');
    bar.insertBefore(btn,prog);
  })(fxx);}
}
function selectHour(fxx){
  currentFxx=fxx;
  document.querySelectorAll('.hbtn').forEach(function(b){b.classList.toggle('active',parseInt(b.dataset.fxx)===fxx);});
  loadData();
}
async function loadData(){
  if(!currentCycle) return;
  var prod=PRODUCTS[currentProduct];
  document.getElementById('load-msg').innerHTML=prod.loadMsg;
  document.getElementById('loading-overlay').classList.remove('hidden');
  document.getElementById('error-bar').style.display='none';
  if(dataLayer){
    if(dataLayer._isStreamline){_slStop(); map.removeLayer(dataLayer);}
    else map.removeLayer(dataLayer);
    dataLayer=null;
  }
  try{
    var url=prod.endpoint+'?fxx='+currentFxx+'&cycle_utc='+encodeURIComponent(currentCycle);
    var resp=await fetch(url);
    if(!resp.ok) throw new Error((await resp.text()).slice(0,200));
    var data=await resp.json();
    renderLayer(data,prod);
    document.getElementById('meta-valid').textContent=data.valid_utc||'—';
    document.getElementById('meta-pts').textContent=(data.point_count||data.points.length).toLocaleString();
  }catch(e){
    var eb=document.getElementById('error-bar'); eb.textContent=e.message; eb.style.display='block';
  }finally{ document.getElementById('loading-overlay').classList.add('hidden'); }
}
function renderLayer(data,prod){
  if(prod.renderMode==='streamline'){
    _slStop();
    var half=(data.cell_size_deg||0.05)/2, halfLon=(data.cell_size_deg||0.05)*1.25;
    var renderer=L.canvas(), rects=[];
    (data.points||[]).forEach(function(p){
      var color=prod.color(p);
      var rect=L.rectangle([[p.lat-half,p.lon-halfLon],[p.lat+half,p.lon+halfLon]],
        {renderer:renderer,color:color,fillColor:color,fillOpacity:currentOpacity,weight:0});
      rect.bindPopup(prod.popup(p),{maxWidth:180}); rects.push(rect);
    });
    dataLayer=L.layerGroup(rects).addTo(map); dataLayer._isStreamline=true;
    _slStartAnimation(data); return;
  }
  var cell=data.cell_size_deg||0.045, half=cell*0.52, halfLon=cell*1.30;
  var renderer=L.canvas(), rects=[];
  data.points.forEach(function(p){
    var color=prod.color(p);
    var rect=L.rectangle([[p.lat-half,p.lon-halfLon],[p.lat+half,p.lon+halfLon]],
      {renderer:renderer,color:color,fillColor:color,fillOpacity:currentOpacity,weight:0});
    rect.bindPopup(prod.popup(p),{maxWidth:200}); rects.push(rect);
  });
  dataLayer=L.layerGroup(rects).addTo(map);
}
(function(){
  var p=new URLSearchParams(window.location.search).get('product');
  if(p&&PRODUCTS[p]){currentProduct=p; document.getElementById('product-sel').value=p;}
})();
updateLegend();
fetchStatus().then(function(){if(currentCycle) loadData();});
setInterval(fetchStatus,300000);

// ── Streamline particle engine ────────────────────────────────────────────────
var _sl={canvas:null,ctx:null,animId:null,data:null,N:1800,age_max:120,speed_scale:0.25,particles:[]};
function _slInterp(flat,cols,gx,gy){
  var x0=Math.floor(gx),y0=Math.floor(gy),x1=x0+1,y1=y0+1,rows=flat.length/cols;
  if(x0<0||y0<0||x1>=cols||y1>=rows) return 0;
  var fx=gx-x0,fy=gy-y0;
  return flat[y0*cols+x0]*(1-fx)*(1-fy)+flat[y0*cols+x1]*fx*(1-fy)+
         flat[y1*cols+x0]*(1-fx)*fy+flat[y1*cols+x1]*fx*fy;
}
function _slLatLonToGrid(lat,lon,d){
  return[(lon-d.lon_min)/(d.lon_max-d.lon_min)*(d.cols-1),
         (lat-d.lat_min)/(d.lat_max-d.lat_min)*(d.rows-1)];
}
function _slRandomParticle(d){
  return{lat:d.lat_min+Math.random()*(d.lat_max-d.lat_min),
         lon:d.lon_min+Math.random()*(d.lon_max-d.lon_min),
         age:Math.floor(Math.random()*80)};
}
function _slInitParticles(d){_sl.particles=[];for(var i=0;i<_sl.N;i++)_sl.particles.push(_slRandomParticle(d));}
function _slStartAnimation(data){
  _sl.data=data; _slInitParticles(data);
  var container=document.getElementById('map');
  var cvs=document.createElement('canvas');
  cvs.style.cssText='position:absolute;top:0;left:0;pointer-events:none;z-index:500;';
  cvs.width=container.offsetWidth; cvs.height=container.offsetHeight;
  container.appendChild(cvs); _sl.canvas=cvs; _sl.ctx=cvs.getContext('2d');
  map.on('move zoom',_slResetOnMove); _slAnimate();
}
function _slResetOnMove(){if(_sl.data)_slInitParticles(_sl.data);}
function _slAnimate(){
  var ctx=_sl.ctx,d=_sl.data; if(!ctx||!d) return;
  ctx.clearRect(0,0,_sl.canvas.width,_sl.canvas.height);
  var zf=Math.pow(2,map.getZoom()-7)*_sl.speed_scale; ctx.lineWidth=2.8;
  var ps=_sl.particles;
  for(var i=0;i<ps.length;i++){
    var p=ps[i]; p.age++;
    var g=_slLatLonToGrid(p.lat,p.lon,d);
    var u=_slInterp(d.u_flat,d.cols,g[0],g[1]),v=_slInterp(d.v_flat,d.cols,g[0],g[1]);
    var spd=Math.sqrt(u*u+v*v);
    if(!p.trail) p.trail=[];
    p.trail.push([p.lat,p.lon]); if(p.trail.length>10) p.trail.shift();
    if(p.trail.length>1){
      var baseAlpha=Math.min(0.40+(spd/18)*0.55,0.95)*currentOpacity;
      for(var t=1;t<p.trail.length;t++){
        var sa=baseAlpha*(t/p.trail.length);
        var ptA=map.latLngToContainerPoint(p.trail[t-1]),ptB=map.latLngToContainerPoint(p.trail[t]);
        ctx.beginPath(); ctx.strokeStyle='rgba(255,255,255,'+sa.toFixed(2)+')';
        ctx.moveTo(ptA.x,ptA.y); ctx.lineTo(ptB.x,ptB.y); ctx.stroke();
      }
    }
    p.lat+=(v/111000)*zf*40;
    p.lon+=(u/(111000*Math.cos(p.lat*Math.PI/180)))*zf*40;
    if(p.age>_sl.age_max||p.lat<d.lat_min||p.lat>d.lat_max||
       p.lon<d.lon_min||p.lon>d.lon_max) ps[i]=_slRandomParticle(d);
  }
  _sl.animId=requestAnimationFrame(_slAnimate);
}
function _slStop(){
  if(_sl.animId){cancelAnimationFrame(_sl.animId);_sl.animId=null;}
  map.off('move zoom',_slResetOnMove);
  if(_sl.canvas&&_sl.canvas.parentNode)_sl.canvas.parentNode.removeChild(_sl.canvas);
  _sl.canvas=_sl.ctx=_sl.data=null; _sl.particles=[];
}
</script>
</body>
</html>"""

CONUS_MAP_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Model Viewer — RAP13 CONUS</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --border:#30363d;
          --text:#e6edf3; --muted:#8b949e; --ac:#f0883e; }
  *,*::before,*::after { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:system-ui,sans-serif;
         height:100dvh; display:flex; flex-direction:column; overflow:hidden; }
  #header { background:var(--panel); border-bottom:1px solid var(--border);
    padding:0.45rem 0.75rem; display:flex; align-items:center;
    gap:0.75rem; flex-wrap:wrap; flex-shrink:0; }
  #header .title { font-weight:700; font-size:0.95rem; white-space:nowrap; }
  #header .subtitle { font-size:0.72rem; color:var(--muted); }
  .nav-link { font-size:0.75rem; color:var(--ac); text-decoration:none;
              padding:0.25rem 0.5rem; border:1px solid var(--ac);
              border-radius:4px; white-space:nowrap; }
  .nav-link:hover { background:rgba(240,136,62,0.1); }
  select { background:var(--bg); color:var(--text); border:1px solid var(--border);
           border-radius:5px; font-size:0.78rem; padding:0.28rem 0.5rem; cursor:pointer; }
  input[type=range] { background:var(--bg); }
  .ctrl-group { display:flex; align-items:center; gap:0.4rem; }
  .ctrl-label { font-size:0.68rem; color:var(--muted); white-space:nowrap; }
  #hour-bar { display:flex; align-items:center; gap:0.3rem; padding:0.3rem 0.75rem;
    background:var(--panel); border-bottom:1px solid var(--border);
    flex-shrink:0; flex-wrap:wrap; }
  .hbtn { font-size:0.68rem; font-weight:600; padding:0.2rem 0.45rem;
    border-radius:4px; border:1px solid var(--border); background:var(--bg);
    color:var(--muted); cursor:pointer; transition:background 0.15s,color 0.15s;
    position:relative; }
  .hbtn.available { color:var(--text); border-color:#444; }
  .hbtn.active { background:var(--ac); color:#000; border-color:var(--ac); }
  .hbtn.unavail { opacity:0.35; cursor:not-allowed; }
  .dot-badge { position:absolute; top:-3px; right:-3px;
               width:6px; height:6px; border-radius:50%; }
  .dot-green { background:#2ecc71; } .dot-grey { background:#555; }
  #progress-bar { height:3px; background:var(--border); flex:1;
                  border-radius:2px; min-width:60px; }
  #progress-fill { height:100%; background:var(--ac); border-radius:2px;
                   transition:width 0.4s; width:0%; }
  #main { flex:1; display:flex; min-height:0; }
  #map  { flex:1; position:relative; }
  #sidebar { width:220px; background:var(--panel); border-left:1px solid var(--border);
             display:flex; flex-direction:column; flex-shrink:0; overflow-y:auto; }
  #legend { padding:0.75rem; }
  .leg-title { font-size:0.72rem; font-weight:700; color:var(--muted);
               margin-bottom:0.5rem; text-transform:uppercase; letter-spacing:0.05em; }
  .leg-row { display:flex; align-items:center; gap:0.55rem; margin:0.22rem 0;
             font-size:0.72rem; }
  .leg-swatch { width:22px; height:12px; border-radius:2px; flex-shrink:0; }
  #cursor-box { padding:0.6rem 0.75rem; border-top:1px solid var(--border);
                font-size:0.72rem; min-height:80px; }
  .cursor-title { font-size:0.65rem; color:var(--muted); text-transform:uppercase;
                  letter-spacing:0.05em; margin-bottom:0.4rem; }
  #cursor-val { font-size:1.4rem; font-weight:700; color:var(--ac); }
  #cursor-pos { font-size:0.65rem; color:var(--muted); margin-top:0.2rem; }
  #meta { padding:0.5rem 0.75rem; font-size:0.68rem; color:var(--muted);
          border-top:1px solid var(--border); }
  #meta b { color:var(--text); }
  #loading-overlay { position:absolute; inset:0; z-index:2000;
    background:rgba(13,17,23,0.88); display:flex; flex-direction:column;
    align-items:center; justify-content:center; gap:1rem; transition:opacity 0.3s; }
  #loading-overlay.hidden { opacity:0; pointer-events:none; }
  .spinner { width:42px; height:42px; border:3px solid var(--border);
             border-top-color:var(--ac); border-radius:50%;
             animation:spin 0.8s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  #load-msg { font-size:0.8rem; color:var(--muted); text-align:center; max-width:260px; }
  #error-bar { display:none; background:#5a1a1a; color:#f9a8a8;
               padding:0.4rem 0.75rem; font-size:0.78rem;
               border-bottom:1px solid #8b2020; flex-shrink:0; }
  .leaflet-control-layers { background:var(--panel)!important;
    border:1px solid var(--border)!important; color:var(--text)!important; font-size:0.75rem; }
  .leaflet-control-layers label { color:var(--text)!important; }
  /* Cursor crosshair tooltip */
  #cursor-tooltip { position:absolute; pointer-events:none; z-index:1500;
    background:rgba(13,17,23,0.90); border:1px solid var(--border);
    border-radius:5px; padding:0.3rem 0.55rem; font-size:0.72rem;
    white-space:nowrap; display:none; }
</style>
</head>
<body>

<div id="header">
  <span class="title">🌎 RAP13 CONUS</span>
  <span class="subtitle">Wind Gusts — Surface</span>
  <div class="ctrl-group">
    <span class="ctrl-label">CYCLE</span>
    <select id="cycle-sel" onchange="onCycleChange()"><option value="">—</option></select>
  </div>
  <div class="ctrl-group" style="margin-left:auto;">
    <span class="ctrl-label">OPACITY</span>
    <input type="range" id="opacity-slider" min="10" max="100" step="5" value="70"
      style="width:80px;" oninput="updateOpacity(this.value)"/>
    <span id="opacity-val" style="font-size:0.72rem;color:var(--muted);width:28px;">70%</span>
  </div>
  <a class="nav-link" href="/map/hrrr">← HRRR Colorado</a>
</div>

<div id="error-bar"></div>

<div id="hour-bar">
  <span class="ctrl-label">HOUR →</span>
  <div id="progress-bar"><div id="progress-fill"></div></div>
  <span id="cycle-pct" style="font-size:0.68rem;color:var(--muted);white-space:nowrap;"></span>
</div>

<div id="main">
  <div id="map">
    <div id="loading-overlay">
      <div class="spinner"></div>
      <div id="load-msg">Loading RAP13…<br>
        <small style="color:var(--muted)">~30 s first load</small>
      </div>
    </div>
    <div id="cursor-tooltip"></div>
  </div>

  <div id="sidebar">
    <div id="legend">
      <div class="leg-title">Wind Gust (kt)</div>
      <div class="leg-row"><div class="leg-swatch" style="background:#4575b4"></div>&lt; 5 kt — Calm</div>
      <div class="leg-row"><div class="leg-swatch" style="background:#74add1"></div>5–10 kt — Light</div>
      <div class="leg-row"><div class="leg-swatch" style="background:#abd9e9"></div>10–15 kt — Breezy</div>
      <div class="leg-row"><div class="leg-swatch" style="background:#e0f3f8"></div>15–20 kt — Moderate</div>
      <div class="leg-row"><div class="leg-swatch" style="background:#fee090"></div>20–25 kt — Fresh</div>
      <div class="leg-row"><div class="leg-swatch" style="background:#fc8d59"></div>25–35 kt — Strong</div>
      <div class="leg-row"><div class="leg-swatch" style="background:#d73027"></div>35–50 kt — Very Strong</div>
      <div class="leg-row"><div class="leg-swatch" style="background:#a50026"></div>&ge; 50 kt — Extreme</div>
      <div style="margin-top:0.7rem;font-size:0.62rem;color:var(--muted);">
        RAP13 13km grid, stride=2<br>Move cursor over map for value
      </div>
    </div>

    <div id="cursor-box">
      <div class="cursor-title">Cursor Sample</div>
      <div id="cursor-val">—</div>
      <div id="cursor-pos"></div>
    </div>

    <div id="meta">
      <div>Model: <b>RAP13</b></div>
      <div>Valid: <b id="meta-valid">—</b></div>
      <div>Points: <b id="meta-pts">—</b></div>
    </div>
  </div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
// ── Color scale (8-band, diverging blue→red) ─────────────────────────────────
var BANDS = [
  { max:  5, color:'#4575b4' },
  { max: 10, color:'#74add1' },
  { max: 15, color:'#abd9e9' },
  { max: 20, color:'#e0f3f8' },
  { max: 25, color:'#fee090' },
  { max: 35, color:'#fc8d59' },
  { max: 50, color:'#d73027' },
  { max: Infinity, color:'#a50026' },
];
function gustColor(kt){
  for(var i=0;i<BANDS.length;i++) if(kt<BANDS[i].max) return BANDS[i].color;
  return '#a50026';
}

// ── State vars ────────────────────────────────────────────────────────────────
var currentCycle=null, currentFxx=1, currentOpacity=0.70;
var cycleStatus={}, dataLayer=null, pointsFlat=[];

// ── Map setup ─────────────────────────────────────────────────────────────────
var map=L.map('map',{center:[39.5,-98.0],zoom:4,preferCanvas:true});

L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
  {attribution:'Tiles &copy; Esri',maxZoom:10}
).addTo(map);

// ── Boundary layers ───────────────────────────────────────────────────────────
var stateStyle={ color:'#e6edf3', weight:1.8, fill:false, opacity:0.9 };
var artccStyle={ color:'#ffa657', weight:2.0, fill:false, opacity:0.95,
                 dashArray:'8 4' };

var statesLayer=null, artccLayer=null;

// US States GeoJSON
fetch('https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json')
  .then(function(r){return r.json();})
  .then(function(gj){
    statesLayer=L.geoJSON(gj,{style:stateStyle});
    layerControl.addOverlay(statesLayer,'⬜ States');
    statesLayer.addTo(map);   // on by default
  }).catch(function(e){console.warn('States GeoJSON failed',e);});

// ARTCC boundaries — served from local API (tries FAA live, falls back to built-in)
fetch('/api/artcc/boundaries')
  .then(function(r){ if(!r.ok) throw new Error('HTTP '+r.status); return r.json(); })
  .then(function(gj){
    artccLayer=L.geoJSON(gj,{
      style:artccStyle,
      onEachFeature:function(feat,layer){
        var p=feat.properties||{};
        var name=p.NAME||p.IDENT||p.id||p.name||'ARTCC';
        layer.bindTooltip(name,{sticky:true,direction:'center',opacity:0.95});
      }
    });
    layerControl.addOverlay(artccLayer,'🔶 ARTCCs');
    artccLayer.addTo(map);
    bringBoundariesToFront();
    console.log('ARTCC loaded: '+gj.features.length+' centers');
  })
  .catch(function(e){ console.warn('ARTCC load failed:',e); });

// Layer control (populated after GeoJSON loads)
var layerControl=L.control.layers(null,{},
  {collapsed:false,position:'topright'}).addTo(map);

// Roads overlay
var roadsLayer=L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}',
  {attribution:'',maxZoom:10,opacity:0.35});
layerControl.addOverlay(roadsLayer,'≡ Roads');

// ── Cursor sampling ───────────────────────────────────────────────────────────
// Build simple grid index: bucket points by ~1° cell for fast lookup
var _idx={}, _cellSz=1.0;
function _idxKey(lat,lon){
  return Math.floor(lat/_cellSz)+'|'+Math.floor(lon/_cellSz);
}
function buildIndex(pts){
  _idx={};
  for(var i=0;i<pts.length;i++){
    var p=pts[i], k=_idxKey(p.lat,p.lon);
    if(!_idx[k]) _idx[k]=[];
    _idx[k].push(i);
  }
}
function nearestPoint(lat,lon){
  if(!pointsFlat.length) return null;
  var best=null, bestD=1e9;
  // Check 3x3 neighbourhood of cells
  for(var dy=-1;dy<=1;dy++){
    for(var dx=-1;dx<=1;dx++){
      var k=(Math.floor(lat/_cellSz)+dy)+'|'+(Math.floor(lon/_cellSz)+dx);
      var cell=_idx[k];
      if(!cell) continue;
      for(var i=0;i<cell.length;i++){
        var p=pointsFlat[cell[i]];
        var d=(p.lat-lat)*(p.lat-lat)+(p.lon-lon)*(p.lon-lon);
        if(d<bestD){bestD=d;best=p;}
      }
    }
  }
  return bestD<4 ? best : null;   // only return if within ~2° (~200km)
}

var tooltip=document.getElementById('cursor-tooltip');
map.on('mousemove',function(e){
  var p=nearestPoint(e.latlng.lat,e.latlng.lng);
  if(p){
    var color=gustColor(p.gust_kt);
    // Sidebar panel
    document.getElementById('cursor-val').textContent=p.gust_kt.toFixed(0)+' kt';
    document.getElementById('cursor-val').style.color=color;
    document.getElementById('cursor-pos').textContent=
      p.lat.toFixed(2)+'°N  '+Math.abs(p.lon).toFixed(2)+'°W';
    // Floating tooltip near cursor
    tooltip.style.display='block';
    var cp=map.latLngToContainerPoint(e.latlng);
    tooltip.style.left=(cp.x+18)+'px';
    tooltip.style.top=(cp.y-12)+'px';
    tooltip.innerHTML='<span style="color:'+color+';font-weight:700;">'+
      p.gust_kt.toFixed(0)+' kt</span>';
  } else {
    tooltip.style.display='none';
    document.getElementById('cursor-val').textContent='—';
    document.getElementById('cursor-val').style.color='var(--ac)';
    document.getElementById('cursor-pos').textContent='';
  }
});
map.on('mouseout',function(){
  tooltip.style.display='none';
  document.getElementById('cursor-val').textContent='—';
  document.getElementById('cursor-pos').textContent='';
});

// ── Opacity ───────────────────────────────────────────────────────────────────
function updateOpacity(val){
  currentOpacity=val/100;
  document.getElementById('opacity-val').textContent=val+'%';
  if(dataLayer && dataLayer.setOpacity) dataLayer.setOpacity(currentOpacity);
}

// ── Cycle status ──────────────────────────────────────────────────────────────
async function fetchStatus(){
  try{
    var s=await(await fetch('/api/rap/status')).json();
    cycleStatus={};
    (s.cycles||[]).forEach(function(c){cycleStatus[c.cycle_utc]=c;});
    var sel=document.getElementById('cycle-sel'), prev=sel.value;
    sel.innerHTML='';
    Object.keys(cycleStatus).sort().reverse().forEach(function(c){
      var opt=document.createElement('option'); opt.value=c;
      opt.textContent='RAP '+new Date(c).toUTCString().slice(5,22)+'Z';
      sel.appendChild(opt);
    });
    if(prev&&cycleStatus[prev]) sel.value=prev;
    else if(!currentCycle&&sel.options.length){
      sel.value=sel.options[0].value; currentCycle=sel.value;
    }
    buildHourButtons();
    var cs=cycleStatus[currentCycle];
    if(cs){
      document.getElementById('progress-fill').style.width=cs.pct_complete+'%';
      document.getElementById('cycle-pct').textContent=cs.pct_complete+'% ready';
    }
  }catch(e){console.warn('status fetch failed',e);}
}

function onCycleChange(){
  currentCycle=document.getElementById('cycle-sel').value;
  buildHourButtons(); loadData();
}

function buildHourButtons(){
  document.querySelectorAll('.hbtn').forEach(function(b){b.remove();});
  var cs=cycleStatus[currentCycle], avail=cs?cs.available_hours:[];
  var cached=cs?(cs.cached_hours.gusts||[]):[];
  var bar=document.getElementById('hour-bar'), prog=document.getElementById('progress-bar');
  for(var fxx=1;fxx<=18;fxx++){(function(f){
    var btn=document.createElement('button');
    btn.className='hbtn';
    btn.textContent='F'+String(f).padStart(2,'0'); btn.dataset.fxx=f;
    var dot=document.createElement('span'); dot.className='dot-badge';
    dot.classList.add(cached.includes(f)?'dot-green':'dot-grey');
    btn.appendChild(dot);
    if(avail.includes(f)){
      btn.classList.add('available');
      btn.onclick=function(){selectHour(f);};
    } else { btn.classList.add('unavail'); btn.disabled=true; }
    if(f===currentFxx) btn.classList.add('active');
    bar.insertBefore(btn,prog);
  })(fxx);}
}

function selectHour(fxx){
  currentFxx=fxx;
  document.querySelectorAll('.hbtn').forEach(function(b){
    b.classList.toggle('active',parseInt(b.dataset.fxx)===fxx);
  });
  loadData();
}

// ── Data load + render ────────────────────────────────────────────────────────
// ── Image bounds match LAT_MIN/LAT_MAX/LON_MIN/LON_MAX in rap_conus.py
var IMG_BOUNDS = [[22.0, -126.0], [52.0, -64.0]];

function bringBoundariesToFront(){
  if(statesLayer && map.hasLayer(statesLayer)) statesLayer.bringToFront();
  if(artccLayer  && map.hasLayer(artccLayer))  artccLayer.bringToFront();
}

async function loadData(){
  if(!currentCycle) return;
  document.getElementById('loading-overlay').classList.remove('hidden');
  document.getElementById('error-bar').style.display='none';
  if(dataLayer){ map.removeLayer(dataLayer); dataLayer=null; }
  pointsFlat=[];

  try{
    // Fetch point data for cursor sampling
    var ptUrl='/api/rap/conus?fxx='+currentFxx+'&cycle_utc='+encodeURIComponent(currentCycle);
    var ptResp=await fetch(ptUrl);
    if(!ptResp.ok) throw new Error((await ptResp.text()).slice(0,300));
    var ptData=await ptResp.json();
    pointsFlat=ptData.points;
    buildIndex(pointsFlat);
    document.getElementById('meta-valid').textContent=ptData.valid_utc||'—';
    document.getElementById('meta-pts').textContent=
      (ptData.point_count||ptData.points.length).toLocaleString();

    // Server-rendered PNG — smooth, projection-correct, no gaps
    var imgUrl='/api/rap/conus/image?fxx='+currentFxx+
               '&cycle_utc='+encodeURIComponent(currentCycle)+
               '&_t='+Date.now();
    dataLayer=L.imageOverlay(imgUrl, IMG_BOUNDS, {
      opacity:currentOpacity, interactive:false, zIndex:200
    }).addTo(map);

    dataLayer.on('load', bringBoundariesToFront);
    setTimeout(bringBoundariesToFront, 300);

  }catch(e){
    var eb=document.getElementById('error-bar');
    eb.textContent=e.message; eb.style.display='block';
  }finally{
    document.getElementById('loading-overlay').classList.add('hidden');
  }
}

fetchStatus().then(function(){if(currentCycle) loadData();});
setInterval(fetchStatus,300000);

// ── Arrow key navigation ──────────────────────────────────────────────────────
document.addEventListener('keydown', function(e){
  if(e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
  e.preventDefault();  // stop map from panning
  var cs = cycleStatus[currentCycle];
  if(!cs || !cs.available_hours.length) return;
  var avail = cs.available_hours;
  var idx   = avail.indexOf(currentFxx);
  var next;
  if(e.key === 'ArrowRight') {
    next = idx < avail.length-1 ? avail[idx+1] : null;
  } else {
    next = idx > 0 ? avail[idx-1] : null;
  }
  if(next !== null) selectHour(next);
});
</script>
</body>
</html>"""
