"""Tests for the local flight tile geometry."""

import numpy as np
from tools.ml_models.dataset.geometry import (
    FRAME_H_PX,
    FRAME_W_PX,
    slice_frame,
    stitch_tiles,
    tile_hw,
)


def test_slice_and_stitch_round_trip() -> None:
    """Slicing a frame and stitching the tiles restores the frame."""
    frame = np.arange(FRAME_H_PX * FRAME_W_PX, dtype=np.float32).reshape(
        1, 1, FRAME_H_PX, FRAME_W_PX
    )
    tiles = slice_frame(frame)
    height, width = tile_hw()
    assert tiles.shape == (64, 1, height, width)
    restored = stitch_tiles(tiles)
    np.testing.assert_array_equal(restored, frame[0, 0])
