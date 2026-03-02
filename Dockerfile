FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libeccodes-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Single worker to stay within Railway's memory limit.
# --preload initializes the app before forking (prevents C-library segfaults).
CMD gunicorn app:app \
    --bind 0.0.0.0:$PORT \
    --timeout 180 \
    --workers 1 \
    --preload
