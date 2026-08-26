"""Smoke test confirming the tools package imports."""

import importlib


def test_tools_imports() -> None:
    """The tools package imports without error."""
    assert importlib.import_module("tools") is not None


def test_tools_model_imports() -> None:
    """tools.model imports without torch."""
    assert importlib.import_module("tools.model") is not None


def test_tools_model_train_imports_without_torch() -> None:
    """tools.model.train imports without torch."""
    assert importlib.import_module("tools.model.train") is not None


def test_tools_model_export_imports_without_torch() -> None:
    """tools.model.export imports without torch."""
    assert importlib.import_module("tools.model.export") is not None
