"""Command line for the native band matrix.

Contains:
  - main: list planned cells or check a result table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.original_dataset_analysis.bands import verify_band_order
from tools.original_dataset_analysis.matrix import (
    Cell,
    gsd_cells,
    native_cells,
    require_complete,
)


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
        "--axis",
        choices=("native", "gsd"),
        default="native",
        help="native band matrix, or the 12-band set and RGB at every legal side",
    )
    parser.add_argument(
        "--results",
        type=Path,
        help="JSON list of {task, subset, side_px} objects to check for completeness",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """List native cells, check a result file, or train the native sweep.

    Args:
        argv: Arguments excluding the program name. ``None`` reads the process
            arguments. ``train-native`` dispatches to the sweep.

    Returns:
        int: Zero after listing, a complete table, or a finished sweep.

    Raises:
        ValueError: If ``--results`` omits a planned cell or a description is
            not a Sentinel-2 id.
    """
    incoming = list(sys.argv[1:] if argv is None else argv)
    if incoming and incoming[0] == "train-native":
        from tools.original_dataset_analysis.sweep import train_native

        return train_native(incoming[1:])
    args = _parser().parse_args(incoming)
    order = verify_band_order(args.descriptions)
    expected = gsd_cells(order) if args.axis == "gsd" else native_cells(order)
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
