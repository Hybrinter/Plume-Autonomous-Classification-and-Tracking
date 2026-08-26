"""Minimal logging helper. Lightning handles its own stdout; this module is a
one-liner for CLI entry points that want a plain logger."""

from __future__ import annotations

import logging


def get_logger(name: str = "smoke_detection") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
