"""
storage.py — DigitalOcean Spaces (S3-compatible) PNG cache.

PNGs are stored keyed by (model_id, product_id, cycle_utc, fxx).
Object key format: renders/{model_id}/{product_id}/{safe_cycle}/F{fxx:02d}.png

All functions fail gracefully — a Spaces outage falls through to L3 rendering.
"""
import os
import logging

log = logging.getLogger(__name__)

_SPACES_KEY      = os.environ.get("DO_SPACES_KEY", "")
_SPACES_SECRET   = os.environ.get("DO_SPACES_SECRET", "")
_SPACES_BUCKET   = os.environ.get("DO_SPACES_BUCKET", "model-viewer")
_SPACES_REGION   = os.environ.get("DO_SPACES_REGION", "nyc3")
_SPACES_ENDPOINT = os.environ.get(
    "DO_SPACES_ENDPOINT",
    f"https://{os.environ.get('DO_SPACES_REGION', 'nyc3')}.digitaloceanspaces.com",
)

_client = None


def spaces_available() -> bool:
    """True if Spaces credentials are configured."""
    return bool(_SPACES_KEY and _SPACES_SECRET)


def _get_client():
    global _client
    if _client is None:
        import boto3
        _client = boto3.client(
            "s3",
            region_name=_SPACES_REGION,
            endpoint_url=_SPACES_ENDPOINT,
            aws_access_key_id=_SPACES_KEY,
            aws_secret_access_key=_SPACES_SECRET,
        )
    return _client


def object_key(model_id: str, product_id: str, cycle_utc: str, fxx: int) -> str:
    """Consistent object key — safe for S3/Spaces (no colons or plusses)."""
    safe_cycle = cycle_utc.replace(":", "-").replace("+", "p").replace(" ", "_")
    return f"renders/{model_id}/{product_id}/{safe_cycle}/F{fxx:02d}.png"


def get_png(model_id: str, product_id: str, cycle_utc: str, fxx: int) -> bytes | None:
    """
    Fetch a pre-rendered PNG from Spaces.
    Returns None on cache miss or any error — caller falls through to L3.
    """
    if not spaces_available():
        return None
    try:
        obj = _get_client().get_object(
            Bucket=_SPACES_BUCKET,
            Key=object_key(model_id, product_id, cycle_utc, fxx),
        )
        return obj["Body"].read()
    except Exception as e:
        if "NoSuchKey" not in str(e) and "404" not in str(e):
            log.warning("Spaces get_png error: %s", e)
        return None


def put_png(model_id: str, product_id: str, cycle_utc: str, fxx: int,
            png_bytes: bytes) -> bool:
    """
    Upload a rendered PNG to Spaces.
    Returns True on success — caller records metadata in Postgres.
    """
    if not spaces_available():
        return False
    try:
        _get_client().put_object(
            Bucket=_SPACES_BUCKET,
            Key=object_key(model_id, product_id, cycle_utc, fxx),
            Body=png_bytes,
            ContentType="image/png",
            CacheControl="public, max-age=3600",
        )
        log.debug("Spaces: uploaded %s F%02d", f"{model_id}/{product_id}", fxx)
        return True
    except Exception as e:
        log.warning("Spaces put_png error: %s", e)
        return False
