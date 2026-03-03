diff --git a/app.py b/app.py
index 33a0dcceddee02080efaf749737562c5a05ecc68..b55962627cfb61af974e4abefcbbd2dbb5f9da7d 100644
--- a/app.py
+++ b/app.py
@@ -1,63 +1,98 @@
 """
 app.py — model-viewer Flask application
 Generic routes driven by the ProductRegistry.
 """
 
 import threading
 import traceback
 import logging
 from datetime import datetime, timezone
 
+from pathlib import Path
+
 from flask import Flask, jsonify, Response, send_from_directory, request
+from werkzeug.exceptions import HTTPException
 
 from artcc_boundaries import ensure_artcc_geojson, get_artcc_geojson
 from renderer import (
     find_latest_cycle, get_cycle_status,
     render_png, extract_points,
     IMAGE_CACHE, POINTS_CACHE,
     LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
 )
 import products.definitions   # registers all products as a side-effect
 from products import REGISTRY, registry_json, get_product
 
 logging.basicConfig(level=logging.INFO)
 log = logging.getLogger(__name__)
 
 app = Flask(__name__, static_folder="static", static_url_path="/static")
 
 # Kick off ARTCC boundary download in background
 threading.Thread(target=ensure_artcc_geojson, daemon=True).start()
 
 TTL = 600   # seconds — cache rendered images for 10 min
 
 
 # ── static frontend ───────────────────────────────────────────────────────────
 
+def serve_index_or_fallback():
+    """Serve SPA entrypoint when present; otherwise return a helpful fallback page."""
+    index_path = Path(app.static_folder) / "index.html"
+    if index_path.is_file():
+        return send_from_directory("static", "index.html")
+
+    return Response(
+        """<!doctype html>
+<html lang="en">
+  <head>
+    <meta charset="utf-8" />
+    <meta name="viewport" content="width=device-width,initial-scale=1" />
+    <title>Model Viewer API</title>
+  </head>
+  <body style="font-family: system-ui, sans-serif; margin: 2rem;">
+    <h1>Model Viewer API</h1>
+    <p>The front-end bundle (static/index.html) is not present in this deploy.</p>
+    <p>Available endpoints: <code>/health</code>, <code>/api/products</code>.</p>
+  </body>
+</html>
+""",
+        mimetype="text/html",
+        status=200,
+    )
+
+
 @app.get("/")
 def index():
-    return send_from_directory("static", "index.html")
+    return serve_index_or_fallback()
+
+
+@app.get("/map/conus")
+def map_conus():
+    """Legacy route used by Railway deployments."""
+    return serve_index_or_fallback()
 
 
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
@@ -144,26 +179,28 @@ def api_meta(model_id, product_id, cycle_utc, fxx):
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
 
 
 # ── error handler ─────────────────────────────────────────────────────────────
 
 @app.errorhandler(Exception)
 def handle_exception(e):
+    if isinstance(e, HTTPException):
+        return e
     return Response(traceback.format_exc(), mimetype="text/plain", status=500)
