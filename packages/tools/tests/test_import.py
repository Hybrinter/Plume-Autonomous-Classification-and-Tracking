"""Smoke test confirming the tools package imports."""

import importlib


def test_tools_imports() -> None:
    """The tools package imports without error."""
    assert importlib.import_module("tools") is not None


def test_tools_ml_models_imports() -> None:
    """tools.ml_models imports successfully."""
    assert importlib.import_module("tools.ml_models") is not None


def test_tools_ml_models_train_imports() -> None:
    """tools.ml_models.train.loop imports successfully."""
    assert importlib.import_module("tools.ml_models.train.loop") is not None


def test_tools_ml_models_export_imports() -> None:
    """tools.ml_models.export.onnx imports successfully."""
    assert importlib.import_module("tools.ml_models.export.onnx") is not None


def test_tools_ml_models_recipe_imports() -> None:
    """tools.ml_models.train.recipe imports successfully."""
    assert importlib.import_module("tools.ml_models.train.recipe") is not None


def test_tools_ml_models_eval_imports() -> None:
    """tools.ml_models.analysis.eval imports successfully."""
    assert importlib.import_module("tools.ml_models.analysis.eval") is not None


def test_tools_ml_models_arch_registry_imports() -> None:
    """tools.ml_models.arch.registry imports successfully."""
    assert importlib.import_module("tools.ml_models.arch.registry") is not None


def test_tools_ml_models_plots_report_runs_import() -> None:
    """plots, report, and runs import successfully."""
    assert importlib.import_module("tools.ml_models.analysis.plots") is not None
    assert importlib.import_module("tools.ml_models.analysis.report") is not None
    assert importlib.import_module("tools.ml_models.analysis.runs") is not None
