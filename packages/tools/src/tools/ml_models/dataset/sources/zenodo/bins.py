"""Native Zenodo and fixed target-GSD bins."""

from __future__ import annotations

import math

from tools.ml_models.dataset.raw import BinSpec, GsdPair

EXTENT_M = 1200.0
NATIVE_SIDE = 120

DEFAULT_BINS: tuple[BinSpec, ...] = (
    BinSpec("native10", 10.0, 10.0),
    BinSpec("gsd15", 15.0, 15.0),
    BinSpec("gsd20", 20.0, 20.0),
    BinSpec("gsd25", 25.0, 25.0),
    BinSpec("gsd30", 30.0, 30.0),
    BinSpec("gsd35", 35.0, 35.0),
)


def bin_hw(bin_spec: BinSpec) -> tuple[int, int]:
    """Round a fixed 1200 m ground window to H along-track, W lateral."""
    if not all(
        math.isfinite(value) and value >= 10 for value in (bin_spec.lateral_m, bin_spec.along_m)
    ):
        raise ValueError("Zenodo bins may only retain or coarsen native 10 m data")
    return round(EXTENT_M / bin_spec.along_m), round(EXTENT_M / bin_spec.lateral_m)


def actual_gsd(bin_spec: BinSpec) -> GsdPair:
    """Record metres per output pixel after rounding, not the requested target."""
    height, width = bin_hw(bin_spec)
    if min(height, width) < 1:
        raise ValueError("empty output bin")
    return GsdPair(EXTENT_M / width, EXTENT_M / height)
