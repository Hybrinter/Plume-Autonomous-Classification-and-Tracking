"""Fixed 1.2 km tile grid, reflectance coarsening, and mask rasterization.

Contains:
  - LEGAL_SIDES: pixel sides that divide 1200 m into 10, 15, 20, 30, or 40 m.
  - gsd_m: ground sample distance for a legal side.
  - coarsen: area-weighted resample from the native 120 px grid onto a legal side.
  - resample_area: area-weighted resample onto any side, including 76.
  - rasterize_mask: percentage polygons onto a legal side.
  - rasterize_percent_mask: percentage polygons onto any side, including 76.

76 is a prism proxy side. It is not a member of ``LEGAL_SIDES``.
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


def _require_side(side_px: int) -> None:
    """Raise ValueError unless ``side_px`` is an int >= 1.

    Args:
        side_px: Output side in pixels.

    Returns:
        None.

    Raises:
        ValueError: If ``side_px`` is a bool or is below 1.
    """
    if isinstance(side_px, bool) or not isinstance(side_px, int) or side_px < 1:
        raise ValueError(f"side_px must be an int >= 1; got {side_px!r}")


def _axis_weights(src_len: int, dst_len: int) -> np.ndarray:
    """Return source-pixel overlap weights, shape ``(dst_len, src_len)``.

    Args:
        src_len: Source pixels along one axis.
        dst_len: Destination pixels along the same axis.

    Returns:
        np.ndarray[float64, (dst_len, src_len)]: Overlap of each source pixel
        with each destination bin. Each destination bin spans ``src_len / dst_len``
        source pixels.
    """
    edges = np.linspace(0.0, float(src_len), dst_len + 1)
    source = np.arange(src_len, dtype=np.float64)
    lo = source
    hi = source + 1.0
    left = edges[:-1, None]
    right = edges[1:, None]
    overlap = np.minimum(hi, right) - np.maximum(lo, left)
    clipped: np.ndarray = np.clip(overlap, 0.0, None)
    return clipped


def resample_area(planes: np.ndarray, side_px: int) -> np.ndarray:
    """Area-weight ``planes`` onto a square of ``side_px``.

    Args:
        planes: Float array ``(C, H, W)``. ``H`` and ``W`` are at least 1.
        side_px: Output side. Any int >= 1, including 76.

    Returns:
        np.ndarray[float32, (C, side_px, side_px)]: Area-weighted resample.
        A source that is already ``side_px`` on both axes is copied.

    Raises:
        ValueError: If ``planes`` is not ``(C, H, W)`` with positive axes, or
            ``side_px`` is below 1.
    """
    _require_side(side_px)
    array = np.asarray(planes)
    if array.ndim != 3:
        raise ValueError(f"expected (C, H, W); got {array.shape}")
    channels = int(array.shape[0])
    height = int(array.shape[1])
    width = int(array.shape[2])
    if channels < 1 or height < 1 or width < 1:
        raise ValueError(f"expected positive (C, H, W); got {array.shape}")
    if height == side_px and width == side_px:
        return np.asarray(array, dtype=np.float32).copy()
    row_w = _axis_weights(height, side_px)  # np.ndarray[float64, (side, H)]
    col_w = _axis_weights(width, side_px)  # np.ndarray[float64, (side, W)]
    src = np.asarray(array, dtype=np.float64)  # np.ndarray[float64, (C, H, W)]
    row_sum = row_w.sum(axis=1)
    row_sum = np.where(row_sum == 0.0, 1.0, row_sum)
    col_sum = col_w.sum(axis=1)
    col_sum = np.where(col_sum == 0.0, 1.0, col_sum)
    row_reduced = np.einsum("oh,chw->cow", row_w, src)
    row_reduced /= row_sum[None, :, None]
    out = np.einsum("cow,vw->cov", row_reduced, col_w)
    out /= col_sum[None, None, :]
    result: np.ndarray = out.astype(np.float32)
    return result


def coarsen(planes: np.ndarray, side_px: int) -> np.ndarray:
    """Area-weight a native stack onto a legal ``side_px``.

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
    return resample_area(planes, side_px)


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


def rasterize_percent_mask(
    polygons: Sequence[np.ndarray],
    side_px: int,
    *,
    rule: str = "half",
) -> np.ndarray:
    """Fill percentage-space polygons onto ``side_px``.

    Args:
        polygons: ``(V, 2)`` vertex arrays in percent, x then y.
        side_px: Output side. Any int >= 1, including 76. This function does
            not consult ``LEGAL_SIDES``.
        rule: ``touch`` marks a cell when any sample lies inside a polygon.
            ``half`` marks a cell when at least half of its samples lie inside.
            The default is ``half``.

    Returns:
        np.ndarray: Float32 mask ``(1, side_px, side_px)`` in {0, 1}.

    Raises:
        ValueError: If the side is below 1 or the rule is unknown.
    """
    _require_side(side_px)
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
    return rasterize_percent_mask(polygons, side_px, rule=rule)
