# Model Viewer

HRRR-based aviation weather visualization web application.

## Overview

A Flask application that fetches HRRR (High-Resolution Rapid Refresh) weather model data from NOAA AWS via Herbie and renders it as interactive Leaflet map overlays.

## Stack

- **Backend**: Python 3.12 / Flask / Gunicorn
- **Data**: Herbie + cfgrib for HRRR GRIB2 access (NOAA AWS)
- **Frontend**: Leaflet (in `static/index.html`)
- **Renderer**: matplotlib for PNG generation

## Project Structure

- `app.py` — Flask app with all API routes
- `renderer.py` — shared fetch/render/cache machinery
- `products/` — product registry and definitions
- `static/` — frontend (index.html, artcc.geojson)
- `artcc_boundaries.py` — ARTCC boundary data management
- `*.py` — individual product implementations (froude, icing, llti, virga, winds, winds_surface)

## API Routes

- `GET /` — serves static frontend
- `GET /health` — health check
- `GET /api/products` — list all registered model products
- `GET /api/status/<model_id>/<product_id>` — cycle availability
- `GET /api/image/<model_id>/<product_id>/<cycle_utc>/<fxx>` — rendered PNG overlay
- `GET /api/points/<model_id>/<product_id>/<cycle_utc>/<fxx>` — JSON point data
- `GET /api/barbs/<model_id>/<product_id>/<cycle_utc>/<fxx>` — wind barbs PNG
- `GET /api/meta/<model_id>/<product_id>/<cycle_utc>/<fxx>` — image metadata
- `GET /api/artcc/boundaries` — ARTCC boundary GeoJSON

## Workflow

- **Start application**: `gunicorn app:app --bind 0.0.0.0:5000 --worker-class gthread --workers 1 --threads 4 --timeout 300`

## System Dependencies

- `eccodes` (Nix) — required for cfgrib/pygrib GRIB2 file handling

## Environment Variables (all optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `HERBIE_DATA_DIR` | /tmp/herbie | GRIB download directory |
| `STRICT_PREFLIGHT` | 0 | Fail on preflight compile errors |
