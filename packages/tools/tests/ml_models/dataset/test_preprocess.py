"""Tests for unit conversion and the model GSD encoding."""

import numpy as np
import pytest
from tools.ml_models.dataset.preprocess import (
    dequantize_unit,
    quantize_unit,
    to_model_gsd,
    to_unit,
)


def test_dn_uses_adc_full_scale() -> None:
    """12-bit full scale maps to 1 and a negative DN clips to 0."""
    image = np.array([[[-10]], [[4095]], [[0]]], dtype=np.float32)
    unit = to_unit(image, "dn", 12)
    assert unit[1, 0, 0] == np.float32(1.0)
    assert unit[0, 0, 0] == np.float32(0.0)


def test_unit_domain_clips() -> None:
    """A unit image is clipped to [0, 1]."""
    image = np.array([[[1.5]], [[-0.2]], [[0.25]]], dtype=np.float32)
    unit = to_unit(image, "unit", 12)
    assert unit[0, 0, 0] == np.float32(1.0)
    assert unit[1, 0, 0] == np.float32(0.0)
    assert unit[2, 0, 0] == np.float32(0.25)


def test_uint16_grid_round_trips() -> None:
    """quantize then dequantize returns the stored grid value."""
    unit = np.array([[[0.0]], [[0.5]], [[1.0]]], dtype=np.float32)
    stored = quantize_unit(unit)
    assert stored.dtype == np.uint16
    assert stored[2, 0, 0] == 65535
    recovered = dequantize_unit(stored)
    again = dequantize_unit(quantize_unit(recovered))
    np.testing.assert_array_equal(recovered, again)


def test_reference_gsd_encodes_to_zero() -> None:
    """GSD equal to the reference is the zero vector."""
    encoded = to_model_gsd(np.array([15.87, 15.87], dtype=np.float32), 15.87)
    np.testing.assert_allclose(encoded, np.zeros(2, dtype=np.float32), atol=1e-6)


def test_malformed_gsd_shape_is_rejected() -> None:
    """GSD input must carry a length-2 component axis."""
    with pytest.raises(ValueError, match="gsd metres"):
        to_model_gsd(np.array([15.87, 15.87, 15.87], dtype=np.float32), 15.87)
    with pytest.raises(ValueError, match="gsd metres"):
        to_model_gsd(np.zeros((2, 3), dtype=np.float32), 15.87)
    with pytest.raises(ValueError, match="gsd metres"):
        to_model_gsd(np.empty((0, 2), dtype=np.float32), 15.87)
