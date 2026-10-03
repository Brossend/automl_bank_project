from __future__ import annotations

import logging

from src.config import LOG_FORMAT


def configure_logging(level: int = logging.INFO) -> None:
    """Configure application logging for command-line entry points."""
    logging.basicConfig(level=level, format=LOG_FORMAT)
