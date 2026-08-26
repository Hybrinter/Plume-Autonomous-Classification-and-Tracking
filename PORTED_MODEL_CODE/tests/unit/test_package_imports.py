"""Smoke-level package import + version check (replaces tests/test_placeholder.py)."""

from __future__ import annotations


def test_package_imports():
    import smoke_detection

    assert smoke_detection.__version__
