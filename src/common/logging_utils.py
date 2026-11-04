"""
Logging layer
Format log seragam untuk semua fase agar output di `docker compose logs`
mudah dibaca dan bisa di-grep per komponen.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Kembalikan logger yang sudah terkonfigurasi (idempotent)."""
    global _CONFIGURED

    if not _CONFIGURED:
        from src.common.config import get_settings

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))

        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(get_settings().log_level)

        # Log Spark/py4j sangat berisik pada level INFO -> naikkan ke WARN.
        for noisy in ("py4j", "py4j.java_gateway", "py4j.clientserver", "botocore", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        _CONFIGURED = True

    return logging.getLogger(name)


def log_section(logger: logging.Logger, title: str) -> None:
    """Pemisah visual antar tahap - membantu membaca log job yang panjang."""
    logger.info("=" * 70)
    logger.info(title.upper())
    logger.info("=" * 70)
