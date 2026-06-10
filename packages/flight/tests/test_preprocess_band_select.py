"""Unit tests for flight.payload.preprocess band selection -- select_bands().

Satisfies: REQ-AIML-PREP-001, REQ-AIML-IMAG-001
"""

from __future__ import annotations

# third-party
import numpy as np

# flight types
from flight.libs.types import Err, FaultCode, Ok

# module under test
from flight.payload.preprocess import select_bands

_LAYOUT = ("BLUE", "GREEN", "RED", "NIR")


def _make_planes(n_bands: int = 4, h: int = 8, w: int = 8) -> np.ndarray:
    """Return an (n_bands, H, W) float32 array where plane i has all pixels == float(i)."""
    arr = np.zeros((n_bands, h, w), dtype=np.float32)
    for i in range(n_bands):
        arr[i, :, :] = float(i)
    return arr  # np.ndarray[float32, (n_bands, H, W)]


def test_select_single_band() -> None:
    """select_bands with ('GREEN',) returns shape (1, H, W) with the GREEN plane (index 1)."""
    planes = _make_planes()
    result = select_bands(planes, _LAYOUT, band_names=("GREEN",))
    assert isinstance(result, Ok)
    assert result.value.shape == (1, 8, 8)
    np.testing.assert_array_almost_equal(result.value[0], np.full((8, 8), 1.0))


def test_select_all_bands() -> None:
    """select_bands with all four band names returns shape (4, H, W)."""
    planes = _make_planes()
    result = select_bands(planes, _LAYOUT, band_names=_LAYOUT)
    assert isinstance(result, Ok)
    assert result.value.shape == (4, 8, 8)


def test_select_bands_order_preserved() -> None:
    """select_bands returns channels in the requested order, not the layout order."""
    planes = _make_planes()
    result = select_bands(planes, _LAYOUT, band_names=("NIR", "BLUE"))
    assert isinstance(result, Ok)
    np.testing.assert_array_almost_equal(result.value[0], np.full((8, 8), 3.0))  # NIR -> plane 3
    np.testing.assert_array_almost_equal(result.value[1], np.full((8, 8), 0.0))  # BLUE -> plane 0


def test_select_bands_dtype_preserved() -> None:
    """select_bands returns float32 output when given float32 input."""
    planes = _make_planes()
    result = select_bands(planes, _LAYOUT, band_names=_LAYOUT)
    assert isinstance(result, Ok)
    assert result.value.dtype == np.float32


def test_select_unknown_band_is_frame_malformed() -> None:
    """A requested name absent from the layout returns Err(FRAME_MALFORMED)."""
    planes = _make_planes()
    result = select_bands(planes, _LAYOUT, band_names=("MAGENTA",))
    assert isinstance(result, Err)
    assert result.error == FaultCode.FRAME_MALFORMED


def test_select_plane_count_mismatch_is_frame_malformed() -> None:
    """Plane count disagreeing with the layout returns Err(FRAME_MALFORMED)."""
    planes = _make_planes(n_bands=3)
    result = select_bands(planes, _LAYOUT, band_names=("BLUE",))
    assert isinstance(result, Err)
    assert result.error == FaultCode.FRAME_MALFORMED
