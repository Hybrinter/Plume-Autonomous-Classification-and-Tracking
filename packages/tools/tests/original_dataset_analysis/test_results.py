"""Stub tables list every planned cell and leave the score blank."""

from __future__ import annotations

from pathlib import Path

from tools.original_dataset_analysis.bands import verify_band_order
from tools.original_dataset_analysis.results import write_stub_tables

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
    """The tables name RGB at 10 m and ceiling at 15 m, with an empty score."""
    path = tmp_path / "results.md"
    write_stub_tables(verify_band_order(_DESC), path)
    text = path.read_text(encoding="utf-8")
    assert "| classify | rgb | 120 | 10 | |" in text
    assert "| segment | ceiling | 80 | 15 | |" in text
    assert "| classify | loo_B8A | 120 | 10 | |" in text
    assert "| 0." not in text
