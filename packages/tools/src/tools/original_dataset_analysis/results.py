"""Empty result tables for the band study.

Contains:
  - write_stub_tables: markdown rows with a blank score column.
  - write_filled_tables: native scores filled, ground-sample scores blank.
"""

from __future__ import annotations

from collections.abc import Mapping
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


def write_filled_tables(
    order: BandOrder,
    scores: Mapping[tuple[str, str, int], float],
    path: Path,
) -> None:
    """Write native scores and leave the ground-sample score column blank.

    Args:
        order: Verified band order.
        scores: Headline score keyed by ``(task, subset, side_px)``.
        path: Markdown destination. Parent directories are created.

    A missing native key stays blank. Ground-sample rows are always blank.
    """
    lines = [
        "# Original-dataset results",
        "",
        "The native score is PR-AUC for the classifier and Dice for the segmentor.",
        "",
        "## Native matrix",
        "",
        "| task | subset | side_px | gsd_m | score |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cell in native_cells(order):
        key = (cell.task, cell.subset, cell.side_px)
        score = scores.get(key)
        text = "" if score is None else f"{score:.4f}"
        lines.append(
            f"| {cell.task} | {cell.subset} | {cell.side_px} | {gsd_m(cell.side_px):.0f} | {text} |"
        )
    lines.extend(
        [
            "",
            "## Ground sample distance",
            "",
            "These cells are not trained in the native pass.",
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
