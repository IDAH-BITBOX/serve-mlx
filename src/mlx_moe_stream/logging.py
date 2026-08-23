"""Small structured logging setup used by command-line entry points."""

from __future__ import annotations

import logging


def configure_logging(verbose: bool = False) -> logging.Logger:
    """Configure package logging once and return the package logger."""

    logger = logging.getLogger("mlx_moe_stream")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger

