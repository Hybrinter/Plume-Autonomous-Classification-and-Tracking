"""Command line for the native band matrix.

Contains:
  - main: list planned cells or check a result table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.original_dataset_analysis.bands import verify_band_order
from tools.original_dataset_analysis.matrix import Cell, native_cells, require_complete


def _parser() -> argparse.ArgumentParser:
    """Return the argument parser."""
    parser = argparse.ArgumentParser(prog="original-dataset-analysis")
    parser.add_argument(
        "--descriptions",
        nargs="+",
        required=True,
        help="GeoTIFF band descriptions, in file order",
    )
    parser.add_argument(
        "--results",
        type=Path,
        help="JSON list of {task, subset, side_px} objects to check for completeness",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """List native cells, or refuse an incomplete result file.

    Args:
        argv: Arguments excluding the program name. ``None`` reads the process
            arguments.

    Returns:
        int: Zero after listing or after a complete table.

    Raises:
        ValueError: If ``--results`` omits a planned cell or a description is
            not a Sentinel-2 id.
    """
    args = _parser().parse_args(argv)
    order = verify_band_order(args.descriptions)
    expected = native_cells(order)
    if args.results is None:
        for cell in expected:
            print(f"{cell.task} {cell.subset} {cell.side_px}")
        return 0
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    rows = tuple(
        Cell(task=item["task"], subset=item["subset"], side_px=int(item["side_px"]))
        for item in payload
    )
    require_complete(rows, expected)
    print(f"complete {len(rows)}")
    return 0
