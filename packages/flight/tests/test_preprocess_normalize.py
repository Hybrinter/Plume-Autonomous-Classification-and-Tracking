"""Tests for DN -> [0, 1] normalization."""

import numpy as np
from flight.payload.preprocess import normalize_dn


def test_normalize_scales_by_full_scale() -> None:
    """12-bit full scale (4095) maps to 1.0; zero maps to 0.0."""
    planes = np.array([[[0.0, 4095.0]]], dtype=np.float32)  # np.ndarray[float32, (1, 1, 2)]
    out = normalize_dn(planes, bit_depth=12)
    np.testing.assert_allclose(out, [[[0.0, 1.0]]])
    assert out.dtype == np.float32


def test_normalize_clips_out_of_range() -> None:
    """Dark-subtraction undershoot and overshoot clip to [0, 1]."""
    planes = np.array([[[-10.0, 5000.0]]], dtype=np.float32)  # np.ndarray[float32, (1, 1, 2)]
    out = normalize_dn(planes, bit_depth=12)
    np.testing.assert_allclose(out, [[[0.0, 1.0]]])
