"""Tests for stacking a prism camera buffer into (3, H, W)."""

import numpy as np
from flight.libs.types import Err, FaultCode, Ok
from flight.payload.preprocess import stack_channels


def test_stack_channels_keeps_channel_major() -> None:
    """A (3, H, W) buffer is returned as float32 without moving axes."""
    buffer = np.arange(3 * 2 * 4, dtype=np.uint16).reshape(3, 2, 4)
    result = stack_channels(buffer)
    assert isinstance(result, Ok)
    assert result.value.shape == (3, 2, 4)
    assert result.value.dtype == np.float32
    assert result.value[1, 0, 0] == float(buffer[1, 0, 0])


def test_stack_channels_moves_channel_last() -> None:
    """An (H, W, 3) buffer becomes channel-major."""
    buffer = np.zeros((2, 4, 3), dtype=np.uint16)
    buffer[0, 1, 2] = 7
    result = stack_channels(buffer)
    assert isinstance(result, Ok)
    assert result.value.shape == (3, 2, 4)
    assert result.value[2, 0, 1] == 7.0


def test_stack_channels_rejects_mosaic_plane() -> None:
    """A 2-D buffer is malformed for a three-sensor prism camera."""
    result = stack_channels(np.zeros((4, 6), dtype=np.uint16))
    assert isinstance(result, Err)
    assert result.error == FaultCode.FRAME_MALFORMED
    assert isinstance(stack_channels(np.zeros((2, 4, 2), dtype=np.float32)), Err)
