"""Small structured logging setup used by command-line entry points."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEFAULT_LOG_MAX_BYTES = 32 * 1024 * 1024
DEFAULT_LOG_BACKUP_COUNT = 5

_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def configure_logging(
    verbose: bool = False,
    *,
    log_file: str | Path | None = None,
    max_bytes: int = DEFAULT_LOG_MAX_BYTES,
    backup_count: int = DEFAULT_LOG_BACKUP_COUNT,
) -> logging.Logger:
    """Configure package logging once and return the package logger.

    A long-lived server needs its own rotating file rather than shell
    redirection: redirection grows without bound, and it ties the process to the
    stdio of whatever launched it.
    """

    logger = logging.getLogger("mlx_moe_stream")
    if not any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    ):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(stream_handler)
        logger.propagate = False
    if log_file is not None and not any(
        isinstance(handler, RotatingFileHandler) for handler in logger.handlers
    ):
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger

