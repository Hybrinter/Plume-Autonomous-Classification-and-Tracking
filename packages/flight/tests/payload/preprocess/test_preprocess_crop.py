"""Tests for the ROI crop and back-projection preprocessing functions."""

import numpy as np
from flight.payload.preprocess import backproject_pixel, crop_to_roi


def test_crop_to_roi_returns_requested_size() -> None:
    """crop_to_roi returns an array of the requested output size and float32 dtype."""
    bands = np.zeros((4, 100, 100), dtype=np.float32)  # np.ndarray[float32, (C, H, W)]
    cropped, origin = crop_to_roi(bands, center_px=(50, 50), output_size=(20, 20))
    assert cropped.shape == (4, 20, 20)
    assert cropped.dtype == np.float32
    assert isinstance(origin, tuple)
    assert len(origin) == 2


def test_backproject_pixel_round_trips_with_crop_origin() -> None:
    """backproject_pixel adds the crop origin back at unit scale."""
    bands = np.zeros((4, 100, 100), dtype=np.float32)
    _, origin = crop_to_roi(bands, center_px=(50, 50), output_size=(20, 20))
    full = backproject_pixel(px=(0, 0), crop_origin=origin, scale_factor=1.0)
    assert full == origin
