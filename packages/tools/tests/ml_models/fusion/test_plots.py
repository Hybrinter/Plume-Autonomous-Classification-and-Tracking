"""Tests for the fusion-trade figures."""

from __future__ import annotations

from pathlib import Path

import pytest
from tools.ml_models.fusion.plots import write_seed_plot, write_threshold_plot
from tools.ml_models.fusion.thresholds import ALL_BINS, ThresholdRow


def test_seed_plot_writes_png(tmp_path: Path) -> None:
    """The seed plot writes a PNG and rejects an option with no score."""
    out = tmp_path / "fig" / "seeds.png"
    write_seed_plot({"Early": [0.74, 0.75, 0.76], "Late": [0.75]}, "AP", "Seeds", out)
    assert out.read_bytes()[:4] == b"\x89PNG"
    with pytest.raises(ValueError):
        write_seed_plot({"Early": []}, "AP", "Seeds", out)


def test_threshold_plot_writes_png(tmp_path: Path) -> None:
    """The threshold plot writes a PNG and needs at least one bin row."""
    rows = (
        ThresholdRow("el05", 10, 0.6, 0.65, 0.7),
        ThresholdRow("el45", 10, 0.4, 0.55, 0.6),
        ThresholdRow(ALL_BINS, 20, 0.5, 0.6, 0.65),
    )
    out = tmp_path / "thr.png"
    write_threshold_plot(rows, "Fused", out)
    assert out.read_bytes()[:4] == b"\x89PNG"
    with pytest.raises(ValueError):
        write_threshold_plot(rows[-1:], "Fused", out)
