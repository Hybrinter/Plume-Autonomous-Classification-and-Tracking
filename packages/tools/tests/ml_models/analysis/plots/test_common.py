"""Tests for the shared frozen-bundle render boundary."""

from pathlib import Path

from flight.libs.types import Err
from tools.ml_models.analysis.config import PlotConfig
from tools.ml_models.analysis.plots.common import render_analysis


def test_render_analysis_rejects_missing_summary(tmp_path: Path) -> None:
    """A directory without a verified summary fails closed with no output."""
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    out = tmp_path / "figures"
    result = render_analysis(evidence, PlotConfig(), out)
    assert isinstance(result, Err)
    assert not out.exists()


def test_render_analysis_rejects_unreadable_summary(tmp_path: Path) -> None:
    """A corrupt summary document fails verification before any output."""
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "summary.json").write_bytes(b'{"kind": "bogus"}')
    out = tmp_path / "figures"
    result = render_analysis(evidence, PlotConfig(), out)
    assert isinstance(result, Err)
    assert not out.exists()
