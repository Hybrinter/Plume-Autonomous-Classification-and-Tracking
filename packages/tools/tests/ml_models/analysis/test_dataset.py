"""Tests for the unavailable dataset-analysis boundary."""

from collections.abc import Callable
from pathlib import Path

from flight.libs.types import Err
from tools.ml_models.analysis.config import DatasetAnalysisConfig
from tools.ml_models.analysis.dataset import analyze_dataset


def test_analyze_dataset_returns_unavailable(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """A finished dataset still cannot produce an analysis bundle."""
    dataset = build_synthetic_dataset(tmp_path / "ds", n=9)
    out = tmp_path / "analysis-out"
    result = analyze_dataset(DatasetAnalysisConfig(dataset=str(dataset), out=str(out)))
    assert isinstance(result, Err)
    assert "unavailable" in result.error
    assert not out.exists()
