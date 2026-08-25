"""Tests for 2x2 CFA separation and interleave round-trip."""

import numpy as np
from flight.libs.types import Err, FaultCode, Ok
from flight.payload.preprocess import interleave_bands, separate_bands


def test_separate_bands_extracts_cells() -> None:
    """Each band plane is the strided sample of its row-major 2x2 cell."""
    mosaic = np.arange(16, dtype=np.float32).reshape(4, 4)  # np.ndarray[float32, (4, 4)]
    result = separate_bands(mosaic)
    assert isinstance(result, Ok)
    planes = result.value  # np.ndarray[float32, (4, 2, 2)]
    assert planes.shape == (4, 2, 2)
    np.testing.assert_array_equal(planes[0], mosaic[0::2, 0::2])  # cell (0,0)
    np.testing.assert_array_equal(planes[1], mosaic[0::2, 1::2])  # cell (0,1)
    np.testing.assert_array_equal(planes[2], mosaic[1::2, 0::2])  # cell (1,0)
    np.testing.assert_array_equal(planes[3], mosaic[1::2, 1::2])  # cell (1,1)


def test_separate_bands_rejects_odd_or_non_2d() -> None:
    """Odd dimensions or wrong rank return Err(FRAME_MALFORMED)."""
    odd = np.zeros((5, 4), dtype=np.float32)
    result = separate_bands(odd)
    assert isinstance(result, Err)
    assert result.error == FaultCode.FRAME_MALFORMED
    assert isinstance(separate_bands(np.zeros((4,), dtype=np.float32)), Err)


def test_interleave_is_inverse_of_separate() -> None:
    """interleave_bands(separate_bands(m)) reproduces the mosaic."""
    rng = np.random.default_rng(0)
    mosaic = rng.uniform(0.0, 4095.0, size=(8, 8)).astype(np.float32)
    planes = separate_bands(mosaic)
    assert isinstance(planes, Ok)
    rebuilt = interleave_bands(planes.value)
    assert isinstance(rebuilt, Ok)
    np.testing.assert_array_equal(rebuilt.value, mosaic)
