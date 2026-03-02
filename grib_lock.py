"""
grib_lock.py  –  Single global lock for all HRRR GRIB downloads.
Kept in its own module to avoid circular imports between
prefetch.py, virga.py, and froude.py.
"""
import threading
GRIB_LOCK = threading.Lock()
