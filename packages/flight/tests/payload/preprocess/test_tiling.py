"""Tests for flight frame tiling and stitching."""

import numpy as np
from flight.libs.types import Err, FaultCode, Ok
from flight.payload.preprocess.tiling import slice_frame, stitch_tiles, validate_layout


def test_validate_layout() -> None:
    """The default sensor frame divides into 193 by 258 tiles."""
    result = validate_layout((1544, 2064), (8, 8))
    assert isinstance(result, Ok)
    assert result.value == (193, 258)


def test_slice_stitch_round_trip_preserves_channels() -> None:
    """Row-major slices stitch back into the channel-major full frame."""
    frame = np.arange(1 * 2 * 12 * 20, dtype=np.uint16).reshape(1, 2, 12, 20)
    sliced = slice_frame(frame, (3, 4))
    assert isinstance(sliced, Ok)
    assert sliced.value.shape == (12, 2, 4, 5)
    stitched = stitch_tiles(sliced.value, (3, 4))
    assert isinstance(stitched, Ok)
    assert stitched.value.shape == (2, 12, 20)
    np.testing.assert_array_equal(stitched.value, frame[0])


def test_single_channel_axis_is_preserved() -> None:
    """Stitching does not squeeze a singleton channel dimension."""
    frame = np.ones((1, 1, 8, 8), dtype=np.float32)
    result = slice_frame(frame, (2, 2))
    assert isinstance(result, Ok)
    restored = stitch_tiles(result.value, (2, 2))
    assert isinstance(restored, Ok)
    assert restored.value.shape == (1, 8, 8)


def test_malformed_shapes_return_frame_malformed() -> None:
    """Malformed frame, grid, and tile count return typed errors."""
    assert isinstance(validate_layout((10, 10), (3, 2)), Err)
    assert isinstance(validate_layout((0, 10), (1, 1)), Err)
    assert isinstance(slice_frame(np.zeros((3, 8, 8))), Err)
    assert isinstance(slice_frame(np.zeros((1, 1, 9, 8)), (2, 2)), Err)
    assert isinstance(stitch_tiles(np.zeros((3, 1, 2, 2)), (2, 2)), Err)
    bad = validate_layout((10, 10), (3, 2))
    assert isinstance(bad, Err) and bad.error is FaultCode.FRAME_MALFORMED
