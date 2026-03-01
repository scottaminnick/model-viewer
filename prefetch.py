"""
prefetch.py  –  Background pre-fetcher for all HRRR products
=============================================================
On app startup, a daemon thread quietly downloads and processes
F01-F12 for each product (winds, froude, virga) into their
respective in-memory caches.

The UI polls /api/cache/status to know which hours are ready
so it can show "ready" vs "loading" buttons.

Status values per (product, fxx):
  "pending"     - not yet started
  "loading"     - currently downloading/computing
  "ready"       - in cache and available instantly
  "unavailable" - AWS returned 404 (hour not published yet)
  "error"       - unexpected error
"""

import threading
import time
import logging
from datetime import timezone

from grib_lock import GRIB_LOCK

log = logging.getLogger("prefetch")

MAX_FXX = 12

# Shared status dict  { product: { fxx: status_str } }
_STATUS = {
    "winds":  {fxx: "pending" for fxx in range(1, MAX_FXX + 1)},
    "froude": {fxx: "pending" for fxx in range(1, MAX_FXX + 1)},
    "virga":  {fxx: "pending" for fxx in range(1, MAX_FXX + 1)},
}
_STATUS_LOCK  = threading.Lock()
_CYCLE_IN_USE = {"cycle_utc": None}   # set by first successful fetch


def set_status(product, fxx, value):
    with _STATUS_LOCK:
        _STATUS[product][fxx] = value


def get_all_status():
    with _STATUS_LOCK:
        return {
            "cycle_utc": _CYCLE_IN_USE["cycle_utc"],
            "products":  {p: dict(v) for p, v in _STATUS.items()},
        }


def _fetch_one(product, cycle_utc, fxx):
    """Call the appropriate cached fetcher for one (product, fxx) pair."""
    set_status(product, fxx, "loading")
    try:
        # Acquire global lock — timeout so prefetch never starves user requests.
        # If lock is busy (user request in progress), skip and retry next cycle.
        if not GRIB_LOCK.acquire(timeout=10):
            log.info(f"[prefetch] {product} F{fxx:02d} skipped (lock busy, will retry)")
            set_status(product, fxx, "pending")
            return

        try:
            if product == "winds":
                from winds import get_hrrr_gusts_cached
                get_hrrr_gusts_cached(cycle_utc=cycle_utc, fxx=fxx, ttl_seconds=3600)

            elif product == "froude":
                from froude import get_froude_cached
                get_froude_cached(cycle_utc=cycle_utc, fxx=fxx, ttl_seconds=3600)

            elif product == "virga":
                from virga import get_virga_cached
                get_virga_cached(cycle_utc=cycle_utc, fxx=fxx, ttl_seconds=3600)
        finally:
            GRIB_LOCK.release()

        set_status(product, fxx, "ready")
        log.info(f"[prefetch] {product} F{fxx:02d} ready")

    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ["not found", "did not find", "no such file", "404",
                                   "nomads", "full file", "byte-range"]):
            set_status(product, fxx, "unavailable")
            log.debug(f"[prefetch] {product} F{fxx:02d} unavailable (NOMADS or not on AWS yet)")
        elif "grib_lock timeout" in msg or "lock timeout" in msg:
            set_status(product, fxx, "pending")
            log.debug(f"[prefetch] {product} F{fxx:02d} lock timeout, will retry")
        else:
            set_status(product, fxx, "error")
            log.warning(f"[prefetch] {product} F{fxx:02d} error: {e}")


def _prefetch_loop():
    """
    Main background loop.
    1. Determine the latest HRRR cycle from the winds status API.
    2. Work through F01-F12 × 3 products sequentially.
       (Sequential avoids hammering AWS with parallel 200 MB downloads.)
    3. Sleep 10 minutes, then re-check whether a newer cycle is available
       and fill any hours that came online since last run.
    """
    while True:
        try:
            # Get the latest cycle from the winds status cache
            from winds import get_cycle_status_cached
            status = get_cycle_status_cached(ttl_seconds=300)
            cycle_utc = status["cycles"][0]["cycle_utc"]

            # If cycle changed, reset all statuses
            if cycle_utc != _CYCLE_IN_USE["cycle_utc"]:
                log.info(f"[prefetch] New cycle detected: {cycle_utc}")
                with _STATUS_LOCK:
                    _CYCLE_IN_USE["cycle_utc"] = cycle_utc
                    for product in _STATUS:
                        for fxx in range(1, MAX_FXX + 1):
                            _STATUS[product][fxx] = "pending"

            available_hours = status["cycles"][0]["available_hours"]

            # Process each available hour × each product (skip already-ready ones)
            for fxx in available_hours:
                for product in ["winds", "froude"]:   # virga excluded until stable
                    with _STATUS_LOCK:
                        current = _STATUS[product].get(fxx, "pending")
                    if current in ("ready", "loading"):
                        continue
                    _fetch_one(product, cycle_utc, fxx)

        except Exception as e:
            log.warning(f"[prefetch] Loop error: {e}")

        # Sleep 10 minutes before checking for new hours
        time.sleep(600)


def _delayed_start(delay_seconds):
    """Wait a bit so the app is fully up before hammering AWS."""
    time.sleep(delay_seconds)
    _prefetch_loop()


def start_prefetch_thread(delay_seconds=180):
    """
    Call once at app startup.
    Waits delay_seconds before beginning downloads so the first
    user request can complete without competing for bandwidth/memory.
    """
    t = threading.Thread(
        target=_delayed_start,
        args=(delay_seconds,),
        name="prefetch",
        daemon=True,
    )
    t.start()
    log.info(f"[prefetch] Background pre-fetch thread will start in {delay_seconds}s")
