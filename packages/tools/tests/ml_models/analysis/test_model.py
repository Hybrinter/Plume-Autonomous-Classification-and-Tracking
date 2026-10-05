"""Tests for the unavailable model-analysis boundary."""

from pathlib import Path

from flight.libs.types import Err
from tools.ml_models.analysis.config import ModelAnalysisConfig
from tools.ml_models.analysis.model import analyze_model


def test_analyze_model_returns_unavailable(tmp_path: Path) -> None:
    """A run directory still cannot produce an analysis bundle."""
    run = tmp_path / "run"
    run.mkdir()
    out = tmp_path / "model-analysis"
    result = analyze_model(
        ModelAnalysisConfig(run=str(run), out=str(out), checkpoint="last", final_test=True)
    )
    assert isinstance(result, Err)
    assert "unavailable" in result.error
    assert not out.exists()
