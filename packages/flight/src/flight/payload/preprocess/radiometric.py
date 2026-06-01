"""
pact.preprocessing.radiometric -- Radiometric calibration for PACT raw frames.

Satisfies: REQ-AIML-PREP-002

Applies dark-frame subtraction and flat-field correction to remove sensor-specific
fixed-pattern noise and pixel-response non-uniformity (PRNU) from raw imagery.

Calibration model:
    corrected = (raw - dark_frame) / flat_field

where flat_field has values approximately equal to 1.0 (normalised response map).
Division by flat_field compensates for per-pixel gain variations across the sensor.

If the corrected output contains any NaN or Inf values (e.g. due to a zero flat_field
pixel or upstream NaN in the raw data), apply_calibration() returns
Err(FaultCode.INFERENCE_NAN) without raising an exception.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass

# third-party
import numpy as np

# internal
from flight.libs.types import Err, FaultCode, Ok, Result


@dataclass(frozen=True)
class RadiometricCalibration:
    """Calibration frames for dark-frame subtraction and flat-field correction.

    Both arrays must have the same shape as the raw band array they will be applied to.
    They are loaded once at process startup from calibration files on disk and held in
    memory for the lifetime of the inference process.

    Attributes:
        dark_frame: Per-pixel dark signal estimate. Shape (C, H, W), dtype float32.
                    Subtracted from raw before flat-field division.
        flat_field: Normalised per-pixel response map. Shape (C, H, W), dtype float32.
                    Values should be close to 1.0. Zero values cause NaN in output.
    """

    dark_frame: np.ndarray  # (C, H, W) float32
    flat_field: np.ndarray  # (C, H, W) float32, values ~1.0


def apply_calibration(
    raw: np.ndarray,
    cal: RadiometricCalibration,
) -> Result[np.ndarray, FaultCode]:
    """Apply dark-frame subtraction and flat-field correction to a raw band array.

    Correction formula:
        corrected = (raw - dark_frame) / flat_field

    After correction, the output is checked for NaN and Inf. Any invalid pixel
    (including those caused by a zero flat_field element) causes the function to
    return Err(FaultCode.INFERENCE_NAN). No partial results are returned.

    Args:
        raw: Raw multispectral bands. Shape (C, H, W), dtype float32.
             Must match cal.dark_frame and cal.flat_field in shape.
        cal: RadiometricCalibration holding the dark frame and flat field arrays.

    Returns:
        Ok(corrected) -- np.ndarray[float32, (C, H, W)] with calibration applied.
        Err(FaultCode.INFERENCE_NAN) -- if the corrected output contains any NaN or Inf.
    """
    # Step 1: dark-frame subtraction
    dark_subtracted: np.ndarray = raw - cal.dark_frame  # np.ndarray[float32, (C, H, W)]

    # Step 2: flat-field correction
    # numpy will produce NaN where flat_field == 0.0 (no exception raised).
    corrected: np.ndarray = dark_subtracted / cal.flat_field  # np.ndarray[float32, (C, H, W)]

    # Step 3: validate output -- any NaN or Inf is a fault.
    if not np.isfinite(corrected).all():
        return Err(FaultCode.INFERENCE_NAN)

    return Ok(corrected)
