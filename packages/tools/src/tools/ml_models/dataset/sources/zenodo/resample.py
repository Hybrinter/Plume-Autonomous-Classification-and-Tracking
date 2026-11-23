"""Area-overlap downsampling of channel-major image planes."""

from __future__ import annotations

import numpy as np


def _axis_weights(source: int, target: int) -> np.ndarray:
    edges = np.linspace(0.0, float(source), target + 1)
    pixels = np.arange(source, dtype=np.float64)
    weights = np.maximum(
        0.0,
        np.minimum(edges[1:, None], pixels + 1) - np.maximum(edges[:-1, None], pixels),
    )
    return np.asarray(weights / weights.sum(axis=1, keepdims=True))


def resample_area(planes: np.ndarray, tile_hw: tuple[int, int]) -> np.ndarray:
    """Average source-pixel overlap onto a rectangular output grid.

    No detail is synthesized: upsampling is rejected. Native grids are copied.
    """
    array = np.asarray(planes, dtype=np.float64)
    height, width = tile_hw
    if (
        array.ndim != 3
        or min(array.shape) < 1
        or min(tile_hw) < 1
        or not np.all(np.isfinite(array))
    ):
        raise ValueError("expected finite positive channel-major planes and output size")
    if height > array.shape[1] or width > array.shape[2]:
        raise ValueError("GSD adaptation cannot upsample native data")
    rows = _axis_weights(array.shape[1], height)
    cols = _axis_weights(array.shape[2], width)
    return np.asarray(
        np.einsum("oh,chw,vw->cov", rows, array, cols, optimize=True), dtype=np.float32
    )
