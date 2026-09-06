"""Logging setup shared across the prototype package."""

from __future__ import annotations

import logging
import sys

from src.config import LOGGING_FORMAT


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once with a consistent format.

    Idempotent: repeated calls do not stack duplicate handlers.
    """
    root_logger = logging.getLogger()
    if any(
        isinstance(handler, logging.StreamHandler)
        and handler.stream is sys.stderr
        for handler in root_logger.handlers
    ):
        root_logger.setLevel(level)
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(LOGGING_FORMAT))
    root_logger.addHandler(handler)
    root_logger.setLevel(level)