"""Tests for mosaic-plane calibration: bad-pixel repair then dark/flat correction."""

import numpy as np
from flight.libs.types import Err, FaultCode, Ok
from flight.payload.preprocess import MosaicCalibration, calibrate_mosaic, correct_bad_pixels


def _identity_cal(h: int, w: int) -> MosaicCalibration:
    return MosaicCalibration(
        dark_frame=np.zeros((h, w), dtype=np.float32),
        flat_field=np.ones((h, w), dtype=np.float32),
        bad_pixel_mask=np.zeros((h, w), dtype=bool),
    )


def test_correct_bad_pixels_uses_same_band_neighbors() -> None:
    """A bad pixel is replaced by the mean of its four +/-2 (same CFA cell) neighbors."""
    mosaic = np.zeros((8, 8), dtype=np.float32)
    mosaic[4, 4] = 1000.0  # the bad pixel
    mosaic[2, 4], mosaic[6, 4], mosaic[4, 2], mosaic[4, 6] = 10.0, 20.0, 30.0, 40.0
    mask = np.zeros((8, 8), dtype=bool)
    mask[4, 4] = True
    repaired = correct_bad_pixels(mosaic, mask)
    assert repaired[4, 4] == 25.0  # mean of the four same-band neighbors
    assert repaired[2, 4] == 10.0  # good pixels untouched


def test_calibrate_mosaic_applies_dark_and_flat() -> None:
    """corrected = (repaired - dark) / flat, elementwise on the mosaic plane."""
    mosaic = np.full((4, 4), 100.0, dtype=np.float32)
    cal = MosaicCalibration(
        dark_frame=np.full((4, 4), 20.0, dtype=np.float32),
        flat_field=np.full((4, 4), 2.0, dtype=np.float32),
        bad_pixel_mask=np.zeros((4, 4), dtype=bool),
    )
    result = calibrate_mosaic(mosaic, cal)
    assert isinstance(result, Ok)
    np.testing.assert_allclose(result.value, 40.0)


def test_calibrate_mosaic_shape_mismatch_is_frame_malformed() -> None:
    """A mosaic that does not match the calibration shape returns FRAME_MALFORMED."""
    result = calibrate_mosaic(np.zeros((6, 6), dtype=np.float32), _identity_cal(4, 4))
    assert isinstance(result, Err)
    assert result.error == FaultCode.FRAME_MALFORMED


def test_calibrate_mosaic_nonfinite_is_inference_nan() -> None:
    """A zero flat-field pixel produces Err(INFERENCE_NAN), never NaN output."""
    cal = _identity_cal(4, 4)
    bad_flat = cal.flat_field.copy()
    bad_flat[0, 0] = 0.0
    cal2 = MosaicCalibration(cal.dark_frame, bad_flat, cal.bad_pixel_mask)
    result = calibrate_mosaic(np.ones((4, 4), dtype=np.float32), cal2)
    assert isinstance(result, Err)
    assert result.error == FaultCode.INFERENCE_NAN
