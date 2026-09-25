"""The native matrix lists every leave-one-out band and refuses a gap."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.original_dataset_analysis.bands import verify_band_order
from tools.original_dataset_analysis.cli import main
from tools.original_dataset_analysis.matrix import gsd_cells, native_cells, require_complete
from tools.original_dataset_analysis.plots import write_band_bars

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


def test_native_cells_cover_the_12_band_set() -> None:
    """Both tasks include the 12-band set, RGB, and each 12-band dropout."""
    cells = native_cells(verify_band_order(_DESC))
    names = {cell.subset for cell in cells if cell.task == "classify"}
    assert "s2_12" in names
    assert "rgb" in names
    assert "s2_13" not in names
    assert "loo_B8A" in names
    assert "loo_B10" not in names
    assert len(cells) == 28
    assert all(cell.side_px == 120 for cell in cells)
    assert {cell.task for cell in cells} == {"classify", "segment"}


def test_gsd_axis_is_s2_12_and_rgb_only() -> None:
    """Legal sides cover the 12-band set and RGB, and they omit leave-one-out."""
    cells = gsd_cells(verify_band_order(_DESC))
    sides = {cell.side_px for cell in cells}
    names = {cell.subset for cell in cells}
    assert sides == {120, 80, 60, 40, 30}
    assert names == {"s2_12", "rgb"}
    assert all(cell.task in {"classify", "segment"} for cell in cells)


def test_missing_rgb_is_named(tmp_path: Path) -> None:
    """A result file without the RGB classifier cell names that cell."""
    order = verify_band_order(_DESC)
    expected = native_cells(order)
    rows = [cell for cell in expected if not (cell.task == "classify" and cell.subset == "rgb")]
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            [{"task": cell.task, "subset": cell.subset, "side_px": cell.side_px} for cell in rows]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="subset=rgb"):
        main(["--descriptions", *_DESC, "--results", str(path)])
    with pytest.raises(ValueError, match="subset=rgb"):
        require_complete(rows, expected)


def test_band_bars_write_a_png(tmp_path: Path) -> None:
    """A score table becomes a PNG file."""
    path = tmp_path / "bars.png"
    write_band_bars(("rgb", "s2_12"), (0.4, 0.8), path)
    assert path.is_file()
    assert path.stat().st_size > 0
