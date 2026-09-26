"""Figure writers in tools.ml_models.analysis.plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tools.ml_models.analysis.plots import (
    write_blob_area_histogram,
    write_canvas_preview,
    write_empty_fpr,
    write_gsd_lines,
    write_hit_rate_by_placement,
    write_learning_rate,
    write_logit_margin,
    write_reliability,
    write_score_histogram,
)


def test_canvas_preview_draws_a_feathered_mask(tmp_path: Path) -> None:
    """A tiny canvas and a soft mask border write one PNG."""
    image = np.zeros((3, 8, 8), dtype=np.float32)
    image[:, 2:6, 2:6] = 0.8
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[3:5, 3:5] = 1.0
    mask[2, 2:6] = 0.35
    mask[5, 2:6] = 0.35
    path = write_canvas_preview(image, mask, tmp_path / "canvas.png")
    assert path.is_file()
    assert path.stat().st_size > 0


def test_classifier_and_frame_writers(tmp_path: Path) -> None:
    """Score, reliability, placement, FPR, margin, and area charts each write one PNG."""
    histogram = write_score_histogram([0.1, 0.8, 0.8], tmp_path / "scores.png")
    reliability = write_reliability([0.1, 0.9], [0.0, 1.0], tmp_path / "reliability.png")
    placement = write_hit_rate_by_placement(
        {"center": 0.5, "corner": 1.0, "edge": 0.0},
        tmp_path / "placement.png",
    )
    fpr = write_empty_fpr([0.0, 0.5], [0.4, 0.1], tmp_path / "fpr.png")
    margin = write_logit_margin([1.5, 0.2], [1.0, -0.4], tmp_path / "margin.png")
    areas = write_blob_area_histogram([15, 40, 8], tmp_path / "areas.png")
    rate = write_learning_rate([1, 2], [0.01, 0.001], tmp_path / "lr.png")
    for path in (histogram, reliability, placement, fpr, margin, areas, rate):
        assert path.is_file()
        assert path.stat().st_size > 0


def test_gsd_lines_label_native_dice(tmp_path: Path) -> None:
    """The ground-sample chart writes when both series are complete."""
    scores: dict[tuple[str, str, int], float] = {}
    for task in ("classify", "segment"):
        for subset in ("s2_12", "rgb"):
            for side in (120, 80, 60, 40, 30):
                scores[(task, subset, side)] = 0.5
    path = write_gsd_lines(scores, tmp_path / "gsd.png")
    assert path.name == "gsd.png"
    assert path.is_file()


def test_gsd_lines_reject_a_missing_series(tmp_path: Path) -> None:
    """A missing native-grid point is refused."""
    with pytest.raises(ValueError, match="missing"):
        write_gsd_lines({}, tmp_path / "gsd.png")
