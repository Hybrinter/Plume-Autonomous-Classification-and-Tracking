"""Tests for per-channel calibration: bad-pixel repair then dark/flat correction."""

import numpy as np
from flight.libs.types import Err, FaultCode, Ok
from flight.payload.preprocess import MosaicCalibration, calibrate_mosaic, correct_bad_pixels


def _identity_cal(h: int, w: int) -> MosaicCalibration:
    shape = (3, h, w)
    return MosaicCalibration(
        dark_frame=np.zeros(shape, dtype=np.float32),
        flat_field=np.ones(shape, dtype=np.float32),
        bad_pixel_mask=np.zeros(shape, dtype=bool),
    )


def test_correct_bad_pixels_uses_one_pixel_neighbors() -> None:
    """A bad pixel is replaced by the mean of its four +/-1 same-channel neighbors."""
    planes = np.zeros((3, 8, 8), dtype=np.float32)
    planes[1, 4, 4] = 1000.0
    planes[1, 3, 4], planes[1, 5, 4], planes[1, 4, 3], planes[1, 4, 5] = 10.0, 20.0, 30.0, 40.0
    planes[0, 3, 4] = 999.0
    mask = np.zeros((3, 8, 8), dtype=bool)
    mask[1, 4, 4] = True
    repaired = correct_bad_pixels(planes, mask)
    assert repaired[1, 4, 4] == 25.0
    assert repaired[1, 3, 4] == 10.0
    assert repaired[0, 3, 4] == 999.0


def test_calibrate_mosaic_applies_dark_and_flat() -> None:
    """corrected = (repaired - dark) / flat, elementwise on each channel."""
    planes = np.full((3, 4, 4), 100.0, dtype=np.float32)
    cal = MosaicCalibration(
        dark_frame=np.full((3, 4, 4), 20.0, dtype=np.float32),
        flat_field=np.full((3, 4, 4), 2.0, dtype=np.float32),
        bad_pixel_mask=np.zeros((3, 4, 4), dtype=bool),
    )
    result = calibrate_mosaic(planes, cal)
    assert isinstance(result, Ok)
    np.testing.assert_allclose(result.value, 40.0)


def test_calibrate_mosaic_shape_mismatch_is_frame_malformed() -> None:
    """A stack that does not match the calibration shape returns FRAME_MALFORMED."""
    result = calibrate_mosaic(np.zeros((3, 6, 6), dtype=np.float32), _identity_cal(4, 4))
    assert isinstance(result, Err)
    assert result.error == FaultCode.FRAME_MALFORMED


def test_calibrate_mosaic_nonfinite_is_inference_nan() -> None:
    """A zero flat-field pixel produces Err(INFERENCE_NAN), never NaN output."""
    cal = _identity_cal(4, 4)
    bad_flat = cal.flat_field.copy()
    bad_flat[0, 0, 0] = 0.0
    cal2 = MosaicCalibration(cal.dark_frame, bad_flat, cal.bad_pixel_mask)
    result = calibrate_mosaic(np.ones((3, 4, 4), dtype=np.float32), cal2)
    assert isinstance(result, Err)
    assert result.error == FaultCode.INFERENCE_NAN
