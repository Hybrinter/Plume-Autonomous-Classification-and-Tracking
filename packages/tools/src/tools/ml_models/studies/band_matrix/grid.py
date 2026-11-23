"""Legal square GSD grids for the native band study, not dataset construction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tools.ml_models.dataset.sources.zenodo.annotations import rasterize_percent_mask
from tools.ml_models.dataset.sources.zenodo.bins import EXTENT_M, NATIVE_SIDE
from tools.ml_models.dataset.sources.zenodo.resample import resample_area

__all__ = ["EXTENT_M", "NATIVE_SIDE", "LEGAL_SIDES", "gsd_m", "coarsen", "rasterize_mask"]
LEGAL_SIDES: frozenset[int] = frozenset({120, 80, 60, 40, 30})


def gsd_m(side_px: int) -> float:
    """Return metres per pixel for a legal study side."""
    if side_px not in LEGAL_SIDES:
        raise ValueError("invalid native study side")
    return EXTENT_M / side_px


def coarsen(planes: np.ndarray, side_px: int) -> np.ndarray:
    """Area-average the native 120 square image on one study grid."""
    gsd_m(side_px)
    if planes.ndim != 3 or planes.shape[-2:] != (NATIVE_SIDE, NATIVE_SIDE):
        raise ValueError("study source must be a native 120 square stack")
    return resample_area(planes, (side_px, side_px))


def rasterize_mask(
    polygons: Sequence[np.ndarray],
    side_px: int,
    *,
    rule: str,
) -> np.ndarray:
    """Rasterize polygons through the same source geometry as the dataset adapter."""
    gsd_m(side_px)
    return rasterize_percent_mask(polygons, (side_px, side_px), rule=rule)
