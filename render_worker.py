"""
render_worker.py — Standalone foreground render worker.

Calls _warmup_loop() directly so systemd manages the process lifecycle.
products.definitions is imported first to populate the registry before
the loop starts.
"""
import logging
import db
import products.definitions          # populates REGISTRY as a side-effect
from warmup import _warmup_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

db.init_db()
_warmup_loop()
