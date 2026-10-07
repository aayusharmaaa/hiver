"""Logging configuration shared by all scripts."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: str | int = "INFO") -> None:
    """Configure root logging once, writing to stderr."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format=_FORMAT,
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
