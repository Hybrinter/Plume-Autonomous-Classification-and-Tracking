"""Tests for smoke_detection.common.logging."""

from __future__ import annotations

import logging

from smoke_detection.common.logging import get_logger


def test_get_logger_returns_logger_instance():
    log = get_logger("smoke_detection.test_logging.a")
    assert isinstance(log, logging.Logger)


def test_get_logger_is_idempotent_about_handlers():
    name = "smoke_detection.test_logging.b"
    a = get_logger(name)
    b = get_logger(name)
    assert a is b
    assert len(a.handlers) == 1


def test_get_logger_formatter_includes_name_and_level():
    log = get_logger("smoke_detection.test_logging.c")
    fmt = log.handlers[0].formatter._fmt
    assert "%(levelname)s" in fmt
    assert "%(name)s" in fmt
