"""
python/utils/logging_config.py
─────────────────────────────────────────────────────────────────────────────
Centralised logging configuration for Nautilus BTC.

Call configure_logging() once at process startup (done in main.py).
All sub-modules use logging.getLogger(__name__) — no further setup needed.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging(
    level:    int  = logging.INFO,
    log_file: Path = Path("logs/nautilus.log"),
) -> None:
    """
    Configure root logger with:
      - Console handler  : WARNING+ (keeps dashboard clean)
      - File handler     : INFO+    (full audit trail in logs/)
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        fmt     = "%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
        datefmt = "%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # ── File handler ─────────────────────────────────────────────────────────
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # ── Console handler ───────────────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING)   # suppress INFO in terminal (Textual owns stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Suppress noisy third-party loggers
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("hmmlearn").setLevel(logging.WARNING)
