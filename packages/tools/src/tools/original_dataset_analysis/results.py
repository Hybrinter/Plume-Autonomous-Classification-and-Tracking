"""Empty result tables for the band study.

Contains:
  - write_stub_tables: markdown rows with a blank score column.
"""

from __future__ import annotations

from pathlib import Path

from tools.original_dataset_analysis.bands import BandOrder
from tools.original_dataset_analysis.grid import gsd_m
from tools.original_dataset_analysis.matrix import gsd_cells, native_cells


def write_stub_tables(order: BandOrder, path: Path) -> None:
    """Write the native and ground-sample tables with blank scores.

    Args:
        order: Verified band order.
        path: Markdown destination. Parent directories are created.

    The score column is empty. This function does not train a model.
    """
    lines = [
        "# Original-dataset results",
        "",
        "Scores are blank until a run fills them.",
        "",
        "## Native matrix",
        "",
        "| task | subset | side_px | gsd_m | score |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cell in native_cells(order):
        lines.append(
            f"| {cell.task} | {cell.subset} | {cell.side_px} | {gsd_m(cell.side_px):.0f} | |"
        )
    lines.extend(
        [
            "",
            "## Ground sample distance",
            "",
            "| task | subset | side_px | gsd_m | score |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for cell in gsd_cells(order):
        lines.append(
            f"| {cell.task} | {cell.subset} | {cell.side_px} | {gsd_m(cell.side_px):.0f} | |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
