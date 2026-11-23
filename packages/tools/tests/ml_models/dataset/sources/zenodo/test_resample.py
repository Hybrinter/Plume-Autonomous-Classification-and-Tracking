"""Tests for area-overlap downsampling."""

import numpy as np
import pytest
from tools.ml_models.dataset.sources.zenodo.resample import resample_area


def test_constant_planes_survive_rectangular_resize() -> None:
    """A constant image is conserved under a non-square downsample."""
    planes = np.full((3, 120, 120), 0.4, dtype=np.float32)
    out = resample_area(planes, (30, 48))
    assert out.shape == (3, 30, 48)
    np.testing.assert_allclose(out, 0.4, atol=1e-6)


def test_ramps_keep_axis_orientation() -> None:
    """An x ramp varies along W; a y ramp varies along H."""
    height, width = 120, 120
    stack = np.zeros((2, height, width), dtype=np.float32)
    stack[0] = np.arange(width, dtype=np.float32)[None, :]
    stack[1] = np.arange(height, dtype=np.float32)[:, None]
    out = resample_area(stack, (30, 60))
    assert out.shape == (2, 30, 60)
    x_row = out[0, 15]
    y_col = out[1, :, 30]
    assert np.all(np.diff(x_row) > 0.0)
    assert np.all(np.diff(y_col) > 0.0)
    np.testing.assert_allclose(out[0, :, 0], 0.5, atol=0.5)
    np.testing.assert_allclose(out[1, 0, :], 1.5, atol=0.5)
    # The ramp means equal the source mean exactly.
    np.testing.assert_allclose(out[0].mean(), stack[0].mean(), atol=0.5)
    np.testing.assert_allclose(out[1].mean(), stack[1].mean(), atol=0.5)


def test_native_size_is_an_identity_resample() -> None:
    """Same-size output reproduces the input."""
    stack = np.random.default_rng(0).standard_normal((3, 20, 24)).astype(np.float32)
    np.testing.assert_allclose(resample_area(stack, (20, 24)), stack, atol=1e-6)


def test_upsample_and_bad_input_rejected() -> None:
    """Output must not exceed the source on either axis."""
    planes = np.ones((1, 10, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="upsample"):
        resample_area(planes, (11, 10))
    with pytest.raises(ValueError, match="upsample"):
        resample_area(planes, (10, 12))
    with pytest.raises(ValueError):
        resample_area(np.zeros((10, 10), dtype=np.float32), (5, 5))
    with pytest.raises(ValueError):
        resample_area(planes, (0, 5))
