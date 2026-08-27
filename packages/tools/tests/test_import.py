"""Smoke test confirming the tools package imports."""

import importlib


def test_tools_imports() -> None:
    """The tools package imports without error."""
    assert importlib.import_module("tools") is not None


def test_tools_inference_imports() -> None:
    """tools.inference imports successfully."""
    assert importlib.import_module("tools.inference") is not None


def test_tools_inference_train_imports() -> None:
    """tools.inference.train imports successfully."""
    assert importlib.import_module("tools.inference.train") is not None


def test_tools_inference_export_imports() -> None:
    """tools.inference.export imports successfully."""
    assert importlib.import_module("tools.inference.export") is not None


def test_tools_inference_split_imports() -> None:
    """tools.inference.split imports successfully."""
    assert importlib.import_module("tools.inference.split") is not None


def test_tools_inference_arch_registry_imports_without_torch() -> None:
    """tools.inference.arch.registry imports without torch."""
    assert importlib.import_module("tools.inference.arch.registry") is not None
