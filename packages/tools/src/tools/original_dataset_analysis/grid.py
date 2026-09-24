"""Fixed 1.2 km tile grid, reflectance coarsening, and mask rasterization.

Contains:
  - LEGAL_SIDES: pixel sides that divide 1200 m into 10, 15, 20, 30, or 40 m.
  - gsd_m: ground sample distance for a legal side.
  - coarsen: area-weighted resample from the native 120 px grid.
  - rasterize_mask: percentage polygons onto a legal side.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.path import Path as MplPath

EXTENT_M = 1200.0
NATIVE_SIDE = 120
LEGAL_SIDES: frozenset[int] = frozenset({120, 80, 60, 40, 30})
_PERCENT = 100.0
_SAMPLES = 4


def gsd_m(side_px: int) -> float:
    """Return the ground sample distance in metres for a legal side.

    Args:
        side_px: Output pixels on one side of the 1.2 km tile.

    Returns:
        float: ``1200 / side_px``.

    Raises:
        ValueError: If ``side_px`` is not a legal side.
    """
    if side_px not in LEGAL_SIDES:
        raise ValueError(f"side_px must be one of {sorted(LEGAL_SIDES)}; got {side_px}")
    return EXTENT_M / float(side_px)


def _check_native(planes: np.ndarray) -> None:
    """Raise ValueError unless ``planes`` is ``(C, 120, 120)``."""
    if planes.ndim != 3 or planes.shape[1] != NATIVE_SIDE or planes.shape[2] != NATIVE_SIDE:
        raise ValueError(f"expected (C, 120, 120); got {planes.shape}")


def _weights(side_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Return source-pixel weights for each output row, shape ``(side, 120)``."""
    edges = np.linspace(0.0, float(NATIVE_SIDE), side_px + 1)
    source = np.arange(NATIVE_SIDE, dtype=np.float64)
    lo = source
    hi = source + 1.0
    left = edges[:-1, None]
    right = edges[1:, None]
    overlap = np.minimum(hi, right) - np.maximum(lo, left)
    return np.clip(overlap, 0.0, None), edges


def coarsen(planes: np.ndarray, side_px: int) -> np.ndarray:
    """Area-weight a native stack onto ``side_px``.

    Args:
        planes: Float array ``(C, 120, 120)``.
        side_px: Legal output side.

    Returns:
        np.ndarray: Float32 array ``(C, side_px, side_px)``.

    Raises:
        ValueError: If the source shape or the side is illegal.
    """
    _check_native(np.asarray(planes))
    gsd_m(side_px)
    if side_px == NATIVE_SIDE:
        return np.asarray(planes, dtype=np.float32).copy()
    weights, _edges = _weights(side_px)
    src = np.asarray(planes, dtype=np.float64)
    weight_sum = weights.sum(axis=1)
    weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
    channels, _height, width = src.shape
    row_reduced = np.zeros((channels, side_px, width), dtype=np.float64)
    for out_row in range(side_px):
        row_reduced[:, out_row, :] = np.tensordot(weights[out_row], src, axes=(0, 1))
        row_reduced[:, out_row, :] /= weight_sum[out_row]
    out = np.zeros((channels, side_px, side_px), dtype=np.float64)
    for out_col in range(side_px):
        out[:, :, out_col] = row_reduced @ weights[out_col]
        out[:, :, out_col] /= weight_sum[out_col]
    return out.astype(np.float32)


def _cell_samples(side_px: int) -> np.ndarray:
    """Return percentage-space sample points, shape ``(side, side, S*S, 2)``."""
    step = _PERCENT / float(side_px)
    offsets = (np.arange(_SAMPLES) + 0.5) / float(_SAMPLES)
    points = np.empty((side_px, side_px, _SAMPLES * _SAMPLES, 2), dtype=np.float64)
    for row in range(side_px):
        for col in range(side_px):
            xs = (col + offsets) * step
            ys = (row + offsets) * step
            grid_x, grid_y = np.meshgrid(xs, ys)
            points[row, col, :, 0] = grid_x.ravel()
            points[row, col, :, 1] = grid_y.ravel()
    return points


def rasterize_mask(
    polygons: Sequence[np.ndarray],
    side_px: int,
    *,
    rule: str,
) -> np.ndarray:
    """Fill percentage-space polygons onto one legal side.

    Args:
        polygons: ``(V, 2)`` vertex arrays in percent, x then y.
        side_px: Legal output side.
        rule: ``touch`` marks a cell when any sample lies inside a polygon.
            ``half`` marks a cell when at least half of its samples lie inside.

    Returns:
        np.ndarray: Float32 mask ``(1, side_px, side_px)`` in {0, 1}.

    Raises:
        ValueError: If the side or the rule is illegal.
    """
    gsd_m(side_px)
    if rule not in {"touch", "half"}:
        raise ValueError(f"rule must be 'touch' or 'half'; got {rule!r}")
    mask = np.zeros((side_px, side_px), dtype=np.float32)
    if not polygons:
        return mask[np.newaxis, ...]
    samples = _cell_samples(side_px).reshape(-1, 2)
    inside = np.zeros(samples.shape[0], dtype=bool)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 3:
            continue
        if not np.array_equal(points[0], points[-1]):
            points = np.vstack([points, points[0]])
        inside |= MplPath(points, closed=True).contains_points(samples, radius=1e-6)
    covered = inside.reshape(side_px, side_px, _SAMPLES * _SAMPLES)
    fraction = covered.mean(axis=2)
    if rule == "touch":
        mask[fraction > 0.0] = 1.0
    else:
        mask[fraction >= 0.5] = 1.0
    return mask[np.newaxis, ...]
