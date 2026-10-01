"""Smoke test confirming the tools package imports."""

import importlib


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
    """plots, report, and runs import under the new analysis namespace."""
    assert importlib.import_module("tools.ml_models.analysis.plots") is not None
    assert importlib.import_module("tools.ml_models.analysis.report") is not None
    assert importlib.import_module("tools.ml_models.analysis.runs") is not None


def test_tools_ml_models_studies_import() -> None:
    """The band-matrix study imports under ml_models.studies."""
    assert importlib.import_module("tools.ml_models.studies.band_matrix") is not None


def test_tools_ml_models_zenodo_fetch_imports() -> None:
    """The Zenodo fetch helpers import under dataset.sources.zenodo."""
    assert importlib.import_module("tools.ml_models.dataset.sources.zenodo.fetch") is not None
