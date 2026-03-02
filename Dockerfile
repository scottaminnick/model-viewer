FROM python:3.11-slim

# Install the ecCodes C library that cfgrib/herbie require
RUN apt-get update && \
    apt-get install -y --no-install-recommends libeccodes-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2
