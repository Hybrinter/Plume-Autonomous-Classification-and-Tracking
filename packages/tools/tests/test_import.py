"""Smoke test confirming the tools package imports."""

import importlib

import pytest


def test_tools_imports() -> None:
    """The tools package imports without error."""
    assert importlib.import_module("tools") is not None


def test_tools_ml_models_arch_registry_imports() -> None:
    """tools.ml_models.arch.registry imports successfully."""
    assert importlib.import_module("tools.ml_models.arch.registry") is not None


def test_tools_ml_models_export_imports() -> None:
    """tools.ml_models.export imports without onnx or onnxruntime."""
    assert importlib.import_module("tools.ml_models.export") is not None
    assert importlib.import_module("tools.ml_models.export.precision") is not None


def test_tools_ml_models_analysis_imports() -> None:
    """The evidence-analysis scaffolds import under ml_models.analysis."""
    for name in (
        "tools.ml_models.analysis.contracts",
        "tools.ml_models.analysis.config",
        "tools.ml_models.analysis.artifacts",
        "tools.ml_models.analysis.capture",
        "tools.ml_models.analysis.evaluate",
        "tools.ml_models.analysis.dataset",
        "tools.ml_models.analysis.dataset_artifacts",
        "tools.ml_models.analysis.dataset_figures",
        "tools.ml_models.analysis.dataset_previews",
        "tools.ml_models.analysis.dataset_render",
        "tools.ml_models.analysis.generalization_artifacts",
        "tools.ml_models.analysis.training",
        "tools.ml_models.analysis.model",
        "tools.ml_models.analysis.summaries",
        "tools.ml_models.analysis.metrics",
        "tools.ml_models.analysis.metrics.definitions",
        "tools.ml_models.analysis.metrics.classifier",
        "tools.ml_models.analysis.metrics.segmentation",
        "tools.ml_models.analysis.metrics.calibration",
        "tools.ml_models.analysis.metrics.localization",
        "tools.ml_models.analysis.metrics.boundary",
        "tools.ml_models.analysis.metrics.generalization",
        "tools.ml_models.analysis.plots",
        "tools.ml_models.analysis.plots.common",
        "tools.ml_models.analysis.plots.dataset",
        "tools.ml_models.analysis.plots.training",
        "tools.ml_models.analysis.plots.classifier",
        "tools.ml_models.analysis.plots.segmentation",
        "tools.ml_models.analysis.plots.generalization",
        "tools.ml_models.analysis.visuals",
        "tools.ml_models.analysis.visuals.selection",
        "tools.ml_models.analysis.visuals.dataset",
        "tools.ml_models.analysis.visuals.predictions",
        "tools.ml_models.analysis.cost",
        "tools.ml_models.analysis.runs",
        "tools.ml_models.analysis.pareto",
    ):
        assert importlib.import_module(name) is not None


def test_removed_ml_modules_do_not_import() -> None:
    """Deleted scoring and report modules are absent and old plot APIs are gone."""
    for name in (
        "tools.ml_models.train.metrics",
        "tools.ml_models.train.evaluate",
        "tools.ml_models.analysis.report",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)
    plots = importlib.import_module("tools.ml_models.analysis.plots")
    for removed in ("history_figures", "overlay_figures", "failure_figures", "save_figures"):
        assert not hasattr(plots, removed)


def test_tools_ml_models_studies_import() -> None:
    """The band-matrix study imports under ml_models.studies."""
    assert importlib.import_module("tools.ml_models.studies.band_matrix") is not None


def test_tools_ml_models_zenodo_fetch_imports() -> None:
    """The Zenodo fetch helpers import under dataset.sources.zenodo."""
    assert importlib.import_module("tools.ml_models.dataset.sources.zenodo.fetch") is not None
