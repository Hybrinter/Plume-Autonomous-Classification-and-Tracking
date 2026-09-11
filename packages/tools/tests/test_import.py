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


def test_tools_inference_eval_imports() -> None:
    """tools.inference.eval imports successfully."""
    assert importlib.import_module("tools.inference.eval") is not None


def test_tools_inference_arch_registry_imports() -> None:
    """tools.inference.arch.registry imports successfully."""
    assert importlib.import_module("tools.inference.arch.registry") is not None


def test_tools_inference_plots_report_runs_import() -> None:
    """plots, report, and runs import successfully."""
    assert importlib.import_module("tools.inference.plots") is not None
    assert importlib.import_module("tools.inference.report") is not None
    assert importlib.import_module("tools.inference.runs") is not None
