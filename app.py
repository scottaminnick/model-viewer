"""
app.py — model-viewer Flask application
Generic routes driven by the ProductRegistry.
"""

import os
import threading
import traceback
import logging
import py_compile
import storage
import db
from pathlib import Path
from datetime import datetime, timezone

from flask import Flask, jsonify, Response, send_from_directory, request

from artcc_boundaries import ensure_artcc_geojson, get_artcc_geojson
from renderer import (
    find_latest_cycle, get_cycle_status,
    render_png, extract_points,
    IMAGE_CACHE, POINTS_CACHE,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    render_barbs_png
)
import products.definitions   # registers all products as a side-effect
from products import REGISTRY, registry_json, get_product

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", static_url_path="/static")

def _preflight_compile():
    """Preflight-compile key modules; warn by default, fail only when STRICT_PREFLIGHT=1."""
    strict = os.environ.get("STRICT_PREFLIGHT", "0") == "1"
    for rel in ("llti_threat.py", "virga_threat.py", "products/definitions.py",
                "products/science/sigma_omega.py"):
        path = Path(__file__).with_name(rel) if "/" not in rel else Path(__file__).parent / rel
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError:
            if strict:
                raise
            log.exception("Preflight compile failed for %s; app will continue without strict mode.", path)

_preflight_compile()
db.init_db()
# Kick off ARTCC boundary download in background
threading.Thread(target=ensure_artcc_geojson, daemon=True).start()

TTL = 600   # seconds — cache rendered images for 10 min

# ── static frontend ───────────────────────────────────────────────────────────

@app.get("/")
def index():
    return send_from_directory("static", "index.html")

# ── meta / discovery endpoints ────────────────────────────────────────────────

@app.get("/health")
def health():
    return jsonify(status="ok")

@app.get("/api/products")
def api_products():
    """Return all registered models and products for the UI dropdowns."""
    return jsonify(registry_json())

@app.get("/api/status/<model_id>/<product_id>")
def api_status(model_id, product_id):
    """Cycle availability for a model/product — drives the hour selector."""
    prod = get_product(model_id, product_id)
    # Each product carries its own status cache so they don't collide
    if not hasattr(prod, "_status_cache"):
        prod._status_cache = {}
    cached_keys = IMAGE_CACHE.keys() | POINTS_CACHE.keys()
    status = get_cycle_status(
        model_id        = model_id,
        herbie_model    = prod.herbie_model,
        herbie_product  = prod.herbie_product,
        fxx_max         = prod.fxx_max,
        cached_keys     = cached_keys,
        _status_cache   = prod._status_cache,
    )
    return jsonify(status)

# ── image endpoint  ────────────────────────────────────────────────────────────

@app.get("/api/image/<model_id>/<product_id>/<cycle_utc>/<int:fxx>")
def api_image(model_id, product_id, cycle_utc, fxx):
    # L1 — in-memory TTLCache (sub-ms, session only)
    cached = IMAGE_CACHE.get(model_id, product_id, cycle_utc, fxx, TTL)
    if cached:
        return Response(cached, mimetype="image/png",
                        headers={"Cache-Control": f"public, max-age={TTL}"})

    # L2 — Postgres metadata index + Spaces PNG (persistent across restarts)
    if storage.spaces_available() and db.is_rendered(model_id, product_id, cycle_utc, fxx):
        png = storage.get_png(model_id, product_id, cycle_utc, fxx)
        if png:
            IMAGE_CACHE.set(model_id, product_id, cycle_utc, fxx, png)  # warm L1
            return Response(png, mimetype="image/png",
                            headers={"Cache-Control": f"public, max-age={TTL}"})

    # L3 — full GRIB download + render (authoritative, slow)
    prod = get_product(model_id, product_id)
    cycle_dt = datetime.fromisoformat(cycle_utc).replace(tzinfo=None)
    lat2d, lon2d, vals2d = prod.get_values(cycle_dt, fxx)

    overlay = (prod.get_contour_overlay(cycle_dt, fxx)
               if hasattr(prod, "get_contour_overlay") else None)
    log.info(f"Contour overlay result: {overlay is not None}")

    png = render_png(lat2d, lon2d, vals2d, prod.cmap, prod.norm,
                     prod.render_mode, contour_overlay=overlay)

    # Store in L1 immediately
    IMAGE_CACHE.set(model_id, product_id, cycle_utc, fxx, png)

    # Store in L2 in a background thread — don't block the HTTP response
    if storage.spaces_available():
        key = storage.object_key(model_id, product_id, cycle_utc, fxx)
        png_copy = bytes(png)   # capture for thread closure — don't share a reference
        def _upload_to_l2():
            if storage.put_png(model_id, product_id, cycle_utc, fxx, png_copy):
                db.record_render(model_id, product_id, cycle_utc, fxx, key)
        threading.Thread(target=_upload_to_l2, daemon=True).start()

    return Response(png, mimetype="image/png",
                    headers={"Cache-Control": f"public, max-age={TTL}"})

### `requirements.txt` — add two lines
boto3>=1.35.0
psycopg2-binary>=2.9.10

# ── points endpoint (cursor sampling) ────────────────────────────────────────

@app.get("/api/points/<model_id>/<product_id>/<cycle_utc>/<int:fxx>")
def api_points(model_id, product_id, cycle_utc, fxx):
    """
    Return subsampled grid points for cursor-value display.
    Same data as the image but as JSON {lat, lon, value, ...}.
    """
    cached = POINTS_CACHE.get(model_id, product_id, cycle_utc, fxx, TTL)
    if cached:
        return jsonify(cached)
    prod = get_product(model_id, product_id)
    cycle_dt = datetime.fromisoformat(cycle_utc).replace(tzinfo=None)
    point_vals = prod.get_point_values(cycle_dt, fxx)
    lat2d, lon2d, _ = prod.get_values(cycle_dt, fxx)
    from datetime import timedelta
    valid_dt = (cycle_dt + timedelta(hours=fxx)).replace(tzinfo=timezone.utc)\
               .isoformat(timespec="minutes").replace("+00:00", "Z")
    points = extract_points(lat2d, lon2d, point_vals, prod.stride)
    result = {
        "model_id":    model_id,
        "product_id":  product_id,
        "cycle_utc":   cycle_utc,
        "valid_utc":   valid_dt,
        "fxx":         fxx,
        "units":       prod.units,
        "point_count": len(points),
        "bounds":      [[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
        "points":      points,
    }
    POINTS_CACHE.set(model_id, product_id, cycle_utc, fxx, result)
    return jsonify(result)

@app.get("/api/barbs/<model_id>/<product_id>/<cycle_utc>/<int:fxx>")
def api_barbs(model_id, product_id, cycle_utc, fxx):
    prod = get_product(model_id, product_id)
    if not prod or not prod.supports_barbs:
        return Response("Product does not support barbs", status=404)

    barbs_id = product_id + "_barbs"
    cached = IMAGE_CACHE.get(model_id, barbs_id, cycle_utc, fxx, TTL)
    if cached:
        return Response(cached, mimetype="image/png",
                        headers={"Cache-Control": f"public, max-age={TTL}"})

    cycle_dt = datetime.fromisoformat(cycle_utc).replace(tzinfo=None)
    lat2d, lon2d, u2d, v2d = prod.get_barb_data(cycle_dt, fxx)
    png = render_barbs_png(lat2d, lon2d, u2d, v2d, stride=prod.barb_stride)
    IMAGE_CACHE.set(model_id, barbs_id, cycle_utc, fxx, png)
    return Response(png, mimetype="image/png",
                    headers={"Cache-Control": f"public, max-age={TTL}"})

# ── composite endpoint (sigma-omega 2×2 panel) ───────────────────────────────

@app.get("/api/composite/<model_id>/<product_id>/<cycle_utc>/<int:fxx>")
def api_composite(model_id, product_id, cycle_utc, fxx):
    """Return a composite multi-panel PNG (e.g. sigma-omega 2×2 figure)."""
    cache_id = product_id + "_composite"
    cached = IMAGE_CACHE.get(model_id, cache_id, cycle_utc, fxx, TTL)
    if cached:
        return Response(cached, mimetype="image/png",
                        headers={"Cache-Control": f"public, max-age={TTL}"})

    prod = get_product(model_id, product_id)
    if not hasattr(prod, "get_composite_png"):
        return Response(
            f"Product {model_id}/{product_id} does not support composite rendering",
            status=400,
        )
    cycle_dt = datetime.fromisoformat(cycle_utc).replace(tzinfo=None)
    png = prod.get_composite_png(cycle_dt, fxx)
    IMAGE_CACHE.set(model_id, cache_id, cycle_utc, fxx, png)
    return Response(png, mimetype="image/png",
                    headers={"Cache-Control": f"public, max-age={TTL}"})

# ── image metadata (bounds etc.) ──────────────────────────────────────────────

@app.get("/api/meta/<model_id>/<product_id>/<cycle_utc>/<int:fxx>")
def api_meta(model_id, product_id, cycle_utc, fxx):
    prod = get_product(model_id, product_id)
    cycle_dt = datetime.fromisoformat(cycle_utc).replace(tzinfo=None)
    from datetime import timedelta
    valid_dt = (cycle_dt + timedelta(hours=fxx)).replace(tzinfo=timezone.utc)
    return jsonify({
        "model_id":   model_id,
        "product_id": product_id,
        "label":      prod.label,
        "units":      prod.units,
        "cycle_utc":  cycle_utc,
        "valid_utc":  valid_dt.isoformat(timespec="minutes").replace("+00:00","Z"),
        "fxx":        fxx,
        "bounds":     [[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
        "legend":     prod.legend,
    })


# ── ARTCC boundaries ──────────────────────────────────────────────────────────

@app.get("/api/artcc/boundaries")
def api_artcc():
    data = get_artcc_geojson()
    resp = jsonify(data)
    resp.headers["Cache-Control"] = "public, max-age=3600"
    return resp

@app.get("/api/debug/inventory/<model_id>/<product_id>")
def api_debug_inventory(model_id, product_id):
    """Dump the Herbie inventory for a product — use to find correct search strings."""
    from renderer import find_latest_cycle, HERBIE_DIR
    from herbie import Herbie
    prod = get_product(model_id, product_id)
    cycle_dt = find_latest_cycle(prod.herbie_model, prod.herbie_product)
    save_dir = HERBIE_DIR / f"{model_id}_debug"
    save_dir.mkdir(parents=True, exist_ok=True)
    H = Herbie(cycle_dt, model=prod.herbie_model, product=prod.herbie_product,
               fxx=1, save_dir=str(save_dir), overwrite=False)
    inv = H.inventory()
    return Response(inv.to_string(), mimetype="text/plain")

@app.get("/api/debug/rap-fields")
def debug_rap_fields_route():
    from datetime import datetime, timezone
    from llti_threat import debug_rap_fields
    debug_rap_fields(datetime(2026, 3, 5, 7), 1)
    return "Check Railway logs"

# ── error handler ─────────────────────────────────────────────────────────────

@app.errorhandler(Exception)
def handle_exception(e):
    return Response(traceback.format_exc(), mimetype="text/plain", status=500)
