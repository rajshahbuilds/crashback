"""Project logging conventions.

One place to configure logging so every script/notebook/module emits consistent,
timestamped, level-prefixed lines. Library code should call `get_logger(__name__)`
and never configure handlers itself; entry points (scripts, notebooks) call
`configure_logging()` once.
"""
from __future__ import annotations

import logging

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_configured = False


def configure_logging(level: str | int = "INFO") -> None:
    """Idempotently configure root logging for an entry point."""
    global _configured
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt=_DATE_FORMAT)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger; configures sane defaults if no entry point did."""
    if not _configured:
        configure_logging()
    return logging.getLogger(name)
