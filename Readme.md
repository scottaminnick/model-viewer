# Model Viewer

HRRR-based Colorado aviation weather visualization.

## Products

| Product | Endpoint | Source |
|---------|----------|--------|
| Wind Gusts | `/map/winds` | HRRR sfc — GUST |
| Froude Number | `/map/froude` | HRRR prs — U/V/HGT 700mb + terrain |
| Virga Potential | `/map/virga` | HRRR prs — RH/VVEL multi-level |
| Icing Threat | `/map/icing` | HRRR prs — RH/VVEL 850+700mb |
| Surface Flow | `/map/surface` | HRRR sfc — UGRD/VGRD 10m |
| LLTI | `/map/llti` | HRRR sfc+prs — HPBL-coupled transport wind |

All products available F01–F12 from the latest HRRR cycle.

## API

```
GET /api/<product>/colorado?fxx=<1-12>&cycle_utc=<ISO>
GET /api/winds/status          — cycle availability and cache state
GET /api/cache/status          — prefetch thread status
GET /health                    — ops check
GET /debug/sfc_fields          — dump HRRR sfc GRIB fields
GET /debug/prs_fields          — dump HRRR prs GRIB fields
```

## Stack

- Python 3.11 / Flask / Gunicorn
- Herbie + cfgrib for HRRR GRIB2 access (NOAA AWS)
- Leaflet for interactive mapping
- Background prefetch thread warms F01–F12 cache on startup

## Deploy

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/<you>/model-viewer.git
git push -u origin main
```

Then connect to Railway — it reads `Procfile` and `.python-version` automatically.

## Environment variables (all optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `WINDS_TTL` | 600 | Cache TTL (s) for wind gusts |
| `FROUDE_TTL` | 600 | Cache TTL for Froude |
| `VIRGA_TTL` | 600 | Cache TTL for virga |
| `ICING_TTL` | 600 | Cache TTL for icing |
| `WIND_SURF_TTL` | 600 | Cache TTL for surface wind |
| `LLTI_TTL` | 600 | Cache TTL for LLTI |
| `HERBIE_DATA_DIR` | /tmp/herbie | GRIB download directory |
