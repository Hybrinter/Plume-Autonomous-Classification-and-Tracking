"""Native-resolution band matrix.

Contains:
  - Cell: one task, subset name, and side.
  - native_cells: the 12-band set, RGB, and leave-one-out.
  - require_complete: refuse a table that omits a planned cell.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tools.original_dataset_analysis.bands import BandOrder, BandSpec, resolve_subset
from tools.original_dataset_analysis.grid import LEGAL_SIDES, NATIVE_SIDE

_TASKS: tuple[str, ...] = ("classify", "segment")


@dataclass(frozen=True, slots=True)
class Cell:
    """One planned run.

    Attributes:
        task: ``classify`` or ``segment``.
        subset: Subset name from :func:`resolve_subset`.
        side_px: Output side in pixels.
    """

    task: str
    subset: str
    side_px: int


def native_cells(order: BandOrder) -> tuple[Cell, ...]:
    """Return the native-resolution matrix.

    Args:
        order: Verified band order. Leave-one-out names follow the 12-band set.

    Returns:
        tuple[Cell, ...]: Both tasks for the 12-band set, RGB, and each
        12-band dropout, at 120 pixels. B10 is not a trained input.
    """
    specs = [BandSpec("s2_12"), BandSpec("rgb")]
    full = resolve_subset(order, BandSpec("s2_12"))
    specs.extend(BandSpec("loo", dropped_band=band_id) for band_id in full.ids)
    names = tuple(resolve_subset(order, spec).name for spec in specs)
    return tuple(
        Cell(task=task, subset=name, side_px=NATIVE_SIDE) for task in _TASKS for name in names
    )


def gsd_cells(order: BandOrder) -> tuple[Cell, ...]:
    """Return the 12-band set and RGB cells at every legal side.

    Args:
        order: Verified band order.

    Returns:
        tuple[Cell, ...]: Both tasks for the 12-band set and RGB at sides 120,
        80, 60, 40, and 30. Leave-one-out stays on the native matrix.
    """
    names = (
        resolve_subset(order, BandSpec("s2_12")).name,
        resolve_subset(order, BandSpec("rgb")).name,
    )
    sides = tuple(sorted(LEGAL_SIDES, reverse=True))
    return tuple(
        Cell(task=task, subset=name, side_px=side)
        for task in _TASKS
        for name in names
        for side in sides
    )


def require_complete(rows: Sequence[Cell], expected: Sequence[Cell]) -> None:
    """Refuse a result table that is missing a planned cell.

    Args:
        rows: Cells present in a result table.
        expected: Cells the matrix planned.

    Raises:
        ValueError: If any expected cell is absent. The message names it.
    """
    present = {(row.task, row.subset, row.side_px) for row in rows}
    missing = [cell for cell in expected if (cell.task, cell.subset, cell.side_px) not in present]
    if missing:
        first = missing[0]
        raise ValueError(
            f"missing cell task={first.task} subset={first.subset} side={first.side_px}"
        )
