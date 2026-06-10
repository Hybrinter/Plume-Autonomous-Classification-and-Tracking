"""
flight.payload.preprocess.radiometric -- Radiometric calibration for PACT raw frames.

Satisfies: REQ-AIML-PREP-002

Two calibration paths are provided:

Legacy (C, H, W) path -- RadiometricCalibration + apply_calibration:
    Applies dark-frame subtraction and flat-field correction to a band-stacked
    (C, H, W) float32 array. This path is retained for reference while the
    ingest switchover (Task 7) is completed; it will be removed after Task 7.

Mosaic-plane path (pre-demosaic, physically correct) -- MosaicCalibration + calibrate_mosaic:
    Bad-pixel repair is applied first (correct_bad_pixels), then dark-frame subtraction
    and flat-field correction are applied to the raw (H, W) uint16/float32 mosaic plane
    before CFA separation. This is the physically correct order: calibration artifacts
    are characterised in sensor (mosaic) space.

    Calibration model:
        repaired  = correct_bad_pixels(raw, bad_pixel_mask)
        corrected = (repaired - dark_frame) / flat_field

    If the corrected output contains any NaN or Inf values (e.g. due to a zero
    flat_field pixel), calibrate_mosaic() returns Err(FaultCode.INFERENCE_NAN).
    A shape mismatch between the mosaic and the calibration artifacts returns
    Err(FaultCode.FRAME_MALFORMED).

Calibration artifacts (dark, flat, bad_pixel_mask) for flight are loaded from
checksummed .npy files by flight.payload.calibration_io. The SIL/dev identity
calibration (zero dark, unit flat, no bad pixels) is built by
flight.payload.calibration_io.build_identity_calibration.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass

# third-party
import numpy as np

# internal
from flight.libs.types import Err, FaultCode, Ok, Result


@dataclass(frozen=True, slots=True)
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


# ---------------------------------------------------------------------------
# Mosaic-plane calibration (new path; pre-demosaic, physically correct)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MosaicCalibration:
    """Per-pixel calibration for the RAW mosaic plane (pre-demosaic).

    Loaded once at startup from checksummed artifacts (flight) or built as identity
    (SIL). Applied before CFA separation, where the physics lives: dark signal and
    flat-field response variation are sensor properties characterised in mosaic space.

    Attributes:
        dark_frame: np.ndarray[float32, (H, W)] per-pixel dark signal in DN.
        flat_field: np.ndarray[float32, (H, W)] normalized response map, values ~1.0.
            A zero element causes a non-finite output; calibrate_mosaic catches this.
        bad_pixel_mask: np.ndarray[bool, (H, W)] True where the pixel is unusable.
            Bad pixels are repaired by correct_bad_pixels before dark/flat correction.

    Notes:
        All three arrays must share the same (H, W) shape, matching the sensor mosaic
        dimensions from SensorConfig. Mismatch is caught at calibrate_mosaic call time.
    """

    dark_frame: np.ndarray  # (H, W) float32
    flat_field: np.ndarray  # (H, W) float32, values ~1.0
    bad_pixel_mask: np.ndarray  # (H, W) bool


def correct_bad_pixels(mosaic: np.ndarray, bad_pixel_mask: np.ndarray) -> np.ndarray:
    """Replace bad pixels with the mean of their four same-band (+/-2) neighbors.

    Offsets of +/-2 along each axis stay inside the same 2x2 CFA cell, ensuring the
    replacement uses same-band data and does not mix spectral information. Edge pixels
    use reflected padding (mode="reflect") so boundary bad pixels are also corrected.

    Single-pass: a bad neighbor contributes its raw value. This is acceptable for
    isolated defects; clustered defects should be excluded at sensor characterization
    time (not flagged at runtime).

    Args:
        mosaic: np.ndarray[float32, (H, W)] raw mosaic plane (any numeric dtype
            accepted; output is float32).
        bad_pixel_mask: np.ndarray[bool, (H, W)] True marks pixels to replace.

    Returns:
        np.ndarray[float32, (H, W)] mosaic with bad pixels replaced; good pixels
        are returned unchanged (values cast to float32).
    """
    padded = np.pad(mosaic, 2, mode="reflect")  # np.ndarray[float32, (H+4, W+4)]
    neighbors = (
        padded[:-4, 2:-2] + padded[4:, 2:-2] + padded[2:-2, :-4] + padded[2:-2, 4:]
    ) / 4.0  # np.ndarray[float32, (H, W)]
    return np.where(bad_pixel_mask, neighbors, mosaic).astype(np.float32)


def calibrate_mosaic(
    mosaic: np.ndarray,
    cal: MosaicCalibration,
) -> Result[np.ndarray, FaultCode]:
    """Bad-pixel repair then (repaired - dark) / flat on the raw mosaic plane.

    Applies the physically correct calibration order: bad pixels are interpolated
    first (so they do not pollute the dark/flat statistics), then dark-frame
    subtraction, then flat-field correction. All operations are elementwise on the
    full (H, W) mosaic plane (before CFA separation).

    Args:
        mosaic: np.ndarray[float32, (H, W)] raw mosaic plane. Shape must match
            cal.dark_frame.shape.
        cal: MosaicCalibration with dark_frame, flat_field, and bad_pixel_mask
            all of shape (H, W).

    Returns:
        Ok(np.ndarray[float32, (H, W)]) -- calibrated DN values, all finite.
        Err(FaultCode.FRAME_MALFORMED) -- mosaic.shape != cal.dark_frame.shape.
        Err(FaultCode.INFERENCE_NAN) -- any output pixel is non-finite. This covers a
            division by a zero flat-field element as well as a non-finite value already
            present in the input mosaic.

    Notes:
        Clipping of calibrated values to [0, full_scale] is NOT performed here; that
        is the responsibility of normalize_dn, which clips before scaling to [0, 1].
    """
    if mosaic.shape != cal.dark_frame.shape:
        return Err(FaultCode.FRAME_MALFORMED)
    repaired = correct_bad_pixels(mosaic, cal.bad_pixel_mask)
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected = (repaired - cal.dark_frame) / cal.flat_field  # np.ndarray[float32, (H, W)]
    if not np.isfinite(corrected).all():
        return Err(FaultCode.INFERENCE_NAN)
    return Ok(corrected)
