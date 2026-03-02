FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libeccodes-dev \
        libgdal-dev \
        gdal-bin \
        libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Single worker avoids memory doubling from forking.
# --preload loads the app once before forking so GDAL/geopandas
# don't get re-initialized per worker (avoids C library segfaults).
CMD gunicorn app:app \
    --bind 0.0.0.0:$PORT \
    --timeout 180 \
    --workers 1 \
    --preload
