"""Coarsening and mask alignment on the 1.2 km tile."""

from __future__ import annotations

import numpy as np
import pytest
from tools.original_dataset_analysis.grid import coarsen, gsd_m, rasterize_mask

_CENTER = (np.array([[25.0, 25.0], [75.0, 25.0], [75.0, 75.0], [25.0, 75.0]], dtype=np.float32),)


@pytest.mark.parametrize("side", [120, 80, 60, 40, 30])
def test_legal_side_shapes_and_gsd(side: int) -> None:
    """Each legal side returns aligned image and mask shapes and 1200/side metres."""
    image = np.zeros((2, 120, 120), dtype=np.float32)
    mask = rasterize_mask((), side, rule="half")
    coarse = coarsen(image, side)
    assert coarse.shape == (2, side, side)
    assert mask.shape == (1, side, side)
    assert gsd_m(side) == pytest.approx(1200.0 / side)


def test_illegal_side_and_shape_raise() -> None:
    """Side 256 and a non-native source array are refused."""
    with pytest.raises(ValueError, match="side_px"):
        gsd_m(256)
    with pytest.raises(ValueError, match="120"):
        coarsen(np.zeros((1, 64, 64), dtype=np.float32), 60)


@pytest.mark.parametrize("side", [120, 80])
def test_center_polygon_covers_center_half(side: int) -> None:
    """A polygon over the center 50 percent covers the center half of the mask."""
    mask = rasterize_mask(_CENTER, side, rule="half")[0]
    start = side // 4
    stop = (3 * side) // 4
    assert np.all(mask[start:stop, start:stop] == 1.0)


@pytest.mark.parametrize("side", [120, 80, 60, 40, 30])
def test_hot_pixel_stays_in_its_window(side: int) -> None:
    """A hot source pixel contributes only to output cells whose window overlaps it."""
    image = np.zeros((1, 120, 120), dtype=np.float32)
    image[0, 10, 20] = 120.0
    coarse = coarsen(image, side)
    hits = np.argwhere(coarse[0] > 0.0)
    assert hits.shape[0] >= 1
    scale = 120.0 / float(side)
    for row, col in hits:
        row_lo = float(row) * scale
        row_hi = float(row + 1) * scale
        col_lo = float(col) * scale
        col_hi = float(col + 1) * scale
        assert row_lo < 11.0 and row_hi > 10.0
        assert col_lo < 21.0 and col_hi > 20.0
