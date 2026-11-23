"""Contract checks for the model post-training-analysis boundary."""

from __future__ import annotations

from pathlib import Path

from flight.libs.types import Err
from tools.ml_models.analysis.config import ModelAnalysisConfig
from tools.ml_models.analysis.model import analyze_model


def test_missing_run_reports_actionable_error(tmp_path: Path) -> None:
    """Missing run inputs must return Err, never raise."""
    cfg = ModelAnalysisConfig(run=str(tmp_path / "missing-run"), out=str(tmp_path / "out"))
    result = analyze_model(cfg)
    assert isinstance(result, Err)
    assert "missing-run" in result.error
    assert not (tmp_path / "out").exists()
