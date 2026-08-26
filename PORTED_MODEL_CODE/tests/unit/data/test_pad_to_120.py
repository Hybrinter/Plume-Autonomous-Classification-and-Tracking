"""Tests for SmokePlumeDataset._pad_to_120 helper."""

from __future__ import annotations

import numpy as np

from smoke_detection.data.classification_dataset import _pad_to_120


def test_identity_at_120():
    arr = np.arange(4 * 120 * 120, dtype=np.float32).reshape(4, 120, 120)
    out = _pad_to_120(arr)
    assert out.shape == (4, 120, 120)
    np.testing.assert_array_equal(out, arr)


def test_right_pad_repeats_last_column():
    arr = np.zeros((4, 120, 100), dtype=np.float32)
    arr[..., -1] = 7.0
    out = _pad_to_120(arr)
    assert out.shape == (4, 120, 120)
    # columns 100..119 must all equal the original last column (all 7s)
    assert np.all(out[:, :, 100:] == 7.0)


def test_bottom_pad_repeats_last_row():
    arr = np.zeros((4, 100, 120), dtype=np.float32)
    arr[:, -1, :] = 5.0
    out = _pad_to_120(arr)
    assert out.shape == (4, 120, 120)
    assert np.all(out[:, 100:, :] == 5.0)


def test_both_dimensions_padded():
    arr = np.ones((4, 80, 100), dtype=np.float32)
    out = _pad_to_120(arr)
    assert out.shape == (4, 120, 120)
    assert np.all(out == 1.0)
