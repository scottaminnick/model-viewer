"""
warmup.py — Startup cache warmer for all registered products.

On app startup, a background thread walks every product in the registry,
finds the latest cycle, and pre-renders all forecast hours into Spaces +
Postgres. Subsequent restarts skip already-cached hours entirely.

Skips gracefully if Spaces or Postgres are not configured.
"""

import threading
import time
import logging
from datetime import datetime, timezone

log = logging.getLogger("warmup")

# How long to wait after startup before hammering NOAA AWS.
# Gives gunicorn time to finish booting and serve the first user request.
STARTUP_DELAY_SECONDS = 60

# Forecast hours to warm per product. Adjust if fxx_max varies.
FXX_RANGE = range(1, 13)


def _warm_one(prod, cycle_dt: datetime, fxx: int):
    """
    Render one (product, cycle, fxx) combo and write to Spaces + Postgres.
    Skips if already recorded in Postgres.
    """
    import storage
    import db
    from renderer import render_png, render_barbs_png

    cycle_utc = cycle_dt.replace(tzinfo=timezone.utc).isoformat(
        timespec="minutes"
    ).replace("+00:00", "Z")

    # Fast path — already cached in Postgres, nothing to do
    if db.is_rendered(prod.model_id, prod.product_id, cycle_utc, fxx):
        log.debug("skip %s/%s F%02d — already cached", prod.model_id, prod.product_id, fxx)
        return

    log.info("warming %s/%s %s F%02d", prod.model_id, prod.product_id, cycle_utc, fxx)

    try:
        lat2d, lon2d, vals2d = prod.get_values(cycle_dt, fxx)

        overlay = (
            prod.get_contour_overlay(cycle_dt, fxx)
            if hasattr(prod, "get_contour_overlay")
            else None
        )

        png = render_png(
            lat2d, lon2d, vals2d,
            prod.cmap, prod.norm,
            prod.render_mode,
            contour_overlay=overlay,
        )

        key = storage.object_key(prod.model_id, prod.product_id, cycle_utc, fxx)
        if storage.put_png(prod.model_id, prod.product_id, cycle_utc, fxx, png):
            db.record_render(prod.model_id, prod.product_id, cycle_utc, fxx, key)
            log.info("warmed  %s/%s F%02d", prod.model_id, prod.product_id, fxx)
        else:
            log.warning("Spaces upload failed for %s/%s F%02d",
                        prod.model_id, prod.product_id, fxx)

    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ["not found", "404", "did not find"]):
            log.debug("unavailable: %s/%s F%02d", prod.model_id, prod.product_id, fxx)
        else:
            log.warning("warmup error %s/%s F%02d: %s",
                        prod.model_id, prod.product_id, fxx, e)


def _warmup_loop():
    """
    Main warmup loop. Runs once at startup, then repeats every 30 minutes
    to pick up new cycles as they become available on NOAA AWS.
    """
    import storage
    import db
    from products import REGISTRY
    from renderer import find_latest_cycle

    if not storage.spaces_available():
        log.info("Spaces not configured — warmup disabled.")
        return
    if not db.db_available():
        log.info("Postgres not configured — warmup disabled.")
        return

    while True:
        log.info("warmup: starting pass over all products...")

        for model_id, products in REGISTRY.items():
            for product_id, prod in products.items():

                # Find the latest available cycle for this product
                try:
                    cycle_dt = find_latest_cycle(
                        prod.herbie_model, prod.herbie_product
                    )
                except Exception as e:
                    log.warning("warmup: find_latest_cycle failed for %s/%s: %s",
                                model_id, product_id, e)
                    continue

                fxx_max = min(prod.fxx_max, max(FXX_RANGE))
                for fxx in range(1, fxx_max + 1):
                    _warm_one(prod, cycle_dt, fxx)
                    # Small sleep between renders to avoid starving
                    # user-facing requests for the GRIB lock
                    pause = 10 if "sigma_omega" in product_id else 2
                    time.sleep(pause)

        log.info("warmup: pass complete. sleeping 30 min before next check.")
        time.sleep(1800)


def start_warmup_thread(delay_seconds: int = STARTUP_DELAY_SECONDS):
    """Call once at app startup."""

    def _delayed():
        log.info("warmup: will start in %ds...", delay_seconds)
        time.sleep(delay_seconds)
        _warmup_loop()

    t = threading.Thread(target=_delayed, name="warmup", daemon=True)
    t.start()
    log.info("warmup: background thread scheduled.")
