"""Smoke test confirming the tools package imports."""

import importlib


def test_tools_imports() -> None:
    """The tools package imports without error."""
    assert importlib.import_module("tools") is not None
