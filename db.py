"""
db.py — PostgreSQL metadata index for the render cache.

Tracks which (model, product, cycle, fxx) combos exist in Spaces so
api_image() can do a fast indexed lookup instead of a Spaces GET on
every request.  Falls back gracefully if DATABASE_URL is not set.
"""
import os
import logging

log = logging.getLogger(__name__)

_DATABASE_URL = os.environ.get("DATABASE_URL", "")
_pool = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS render_cache (
    model_id    TEXT    NOT NULL,
    product_id  TEXT    NOT NULL,
    cycle_utc   TEXT    NOT NULL,
    fxx         INTEGER NOT NULL,
    png_key     TEXT    NOT NULL,
    rendered_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (model_id, product_id, cycle_utc, fxx)
);
"""


def db_available() -> bool:
    return bool(_DATABASE_URL)


def _get_pool():
    global _pool
    if _pool is None:
        from psycopg2.pool import ThreadedConnectionPool
        _pool = ThreadedConnectionPool(1, 4, _DATABASE_URL)
    return _pool


def init_db():
    """Create the render_cache table if it doesn't exist. Safe to call repeatedly."""
    if not db_available():
        log.info("db: DATABASE_URL not set — metadata tracking disabled.")
        return
    try:
        pool = _get_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(_SCHEMA)
            conn.commit()
            log.info("db: render_cache table ready.")
        finally:
            pool.putconn(conn)
    except Exception as e:
        log.error("db: init_db failed: %s", e)


def is_rendered(model_id: str, product_id: str, cycle_utc: str, fxx: int) -> bool:
    """
    Single primary-key lookup — ~5ms.
    Returns False on any error so the caller falls through to L3.
    """
    if not db_available():
        return False
    try:
        pool = _get_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT 1 FROM render_cache
                       WHERE model_id=%s AND product_id=%s
                         AND cycle_utc=%s AND fxx=%s""",
                    (model_id, product_id, cycle_utc, fxx),
                )
                return cur.fetchone() is not None
        finally:
            pool.putconn(conn)
    except Exception as e:
        log.warning("db: is_rendered failed: %s", e)
        return False


def record_render(model_id: str, product_id: str, cycle_utc: str,
                  fxx: int, png_key: str):
    """Insert or update a render record (upsert on primary key)."""
    if not db_available():
        return
    try:
        pool = _get_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO render_cache
                           (model_id, product_id, cycle_utc, fxx, png_key)
                       VALUES (%s, %s, %s, %s, %s)
                       ON CONFLICT (model_id, product_id, cycle_utc, fxx)
                       DO UPDATE SET png_key=EXCLUDED.png_key,
                                     rendered_at=NOW()""",
                    (model_id, product_id, cycle_utc, fxx, png_key),
                )
            conn.commit()
        finally:
            pool.putconn(conn)
    except Exception as e:
        log.warning("db: record_render failed: %s", e)
