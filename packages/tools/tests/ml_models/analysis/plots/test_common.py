"""Tests for the unavailable render boundary."""

from pathlib import Path

from flight.libs.types import Err
from tools.ml_models.analysis.config import PlotConfig
from tools.ml_models.analysis.plots.common import render_analysis


def test_render_analysis_returns_unavailable(tmp_path: Path) -> None:
    """A frozen evidence directory still cannot render figures."""
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    out = tmp_path / "figures"
    result = render_analysis(evidence, PlotConfig(), out)
    assert isinstance(result, Err)
    assert "unavailable" in result.error
    assert not out.exists()
