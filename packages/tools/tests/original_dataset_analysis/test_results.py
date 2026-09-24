"""Stub tables list every planned cell and leave the score blank."""

from __future__ import annotations

from pathlib import Path

from tools.original_dataset_analysis.bands import verify_band_order
from tools.original_dataset_analysis.plots import write_loss_curve
from tools.original_dataset_analysis.results import write_filled_tables, write_stub_tables

_DESC = (
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B11",
    "B12",
    "B10",
)


def test_stub_rows_have_blank_scores(tmp_path: Path) -> None:
    """The tables name RGB at 10 m and the 12-band set at 15 m, with an empty score."""
    path = tmp_path / "results.md"
    write_stub_tables(verify_band_order(_DESC), path)
    text = path.read_text(encoding="utf-8")
    assert "| classify | rgb | 120 | 10 | |" in text
    assert "| segment | s2_12 | 80 | 15 | |" in text
    assert "| classify | loo_B8A | 120 | 10 | |" in text
    assert "| 0." not in text


def test_filled_native_score_leaves_ground_sample_blank(tmp_path: Path) -> None:
    """A native score is written and the ground-sample score stays empty."""
    path = tmp_path / "RESULTS.md"
    write_filled_tables(verify_band_order(_DESC), {("classify", "s2_12", 120): 0.5}, path)
    text = path.read_text(encoding="utf-8")
    assert "| classify | s2_12 | 120 | 10 | 0.5000 |" in text
    assert "| segment | s2_12 | 80 | 15 | |" in text


def test_loss_curve_marks_the_selected_epoch(tmp_path: Path) -> None:
    """A log-log chart is written for positive train, validation, and test losses."""
    path = tmp_path / "loss.png"
    write_loss_curve([1, 2], [0.4, 0.2], [0.5, 0.3], [0.6, 0.35], path, selected_epoch=2)
    assert path.is_file()
