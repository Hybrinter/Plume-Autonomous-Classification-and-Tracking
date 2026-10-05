"""Tests for the frozen analysis, evaluation, and plot config records."""

import dataclasses

import pytest
from tools.ml_models.analysis.config import (
    DatasetAnalysisConfig,
    EvaluationConfig,
    ModelAnalysisConfig,
    PlotConfig,
)


def test_dataset_analysis_config_fields() -> None:
    """DatasetAnalysisConfig names the dataset and output directory."""
    cfg = DatasetAnalysisConfig(dataset="ds", out="out")
    assert cfg.dataset == "ds"
    assert cfg.out == "out"
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.out = "other"  # type: ignore[misc]


def test_model_analysis_config_defaults() -> None:
    """ModelAnalysisConfig defaults to the best checkpoint without test."""
    cfg = ModelAnalysisConfig(run="run", out="out")
    assert cfg.checkpoint == "best"
    assert cfg.final_test is False


def test_evaluation_config_defaults() -> None:
    """EvaluationConfig requires kind and split; batch size and device default."""
    cfg = EvaluationConfig(kind="segmentor", split="val")
    assert cfg.batch_size == 2
    assert cfg.device == "cpu"


def test_plot_config_defaults() -> None:
    """PlotConfig defaults to PNG plus SVG at 300 dpi."""
    cfg = PlotConfig()
    assert cfg.formats == ("png", "svg")
    assert cfg.dpi == 300
