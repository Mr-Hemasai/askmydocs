# askmydocs/app/core/logger.py — Shared structured Loguru logger configuration
from __future__ import annotations

import sys
from functools import lru_cache
from typing import Any

from loguru import logger

from app.core.config import settings


@lru_cache(maxsize=1)
def get_logger() -> Any:
    """Return a configured shared Loguru logger."""

    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
    logger.add(
        settings.logs_dir / "askmydocs.log",
        rotation="10 MB",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        enqueue=False,
    )
    return logger
