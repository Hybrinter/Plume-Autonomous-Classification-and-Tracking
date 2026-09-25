"""
flight.payload.preprocess.radiometric -- Per-channel radiometric calibration.

Satisfies: REQ-AIML-PREP-002

Calibration runs on the stacked ``(3, H, W)`` prism planes. Dark signal and
flat-field response are per-channel sensor properties.

Pipeline (MosaicCalibration + correct_bad_pixels + calibrate_mosaic):
    repaired  = correct_bad_pixels(planes, bad_pixel_mask)
    corrected = (repaired - dark_frame) / flat_field

    Bad-pixel repair is applied first so defects do not pollute the dark/flat
    statistics. If the corrected output contains any NaN or Inf (e.g. a zero flat_field
    pixel), calibrate_mosaic() returns Err(FaultCode.INFERENCE_NAN). A shape mismatch
    between the stack and the calibration artifacts returns Err(FaultCode.FRAME_MALFORMED).

Calibration artifacts (dark, flat, bad_pixel_mask) for flight are loaded from
checksummed .npy files by flight.payload.calibration_io. The SIL/dev identity
calibration (zero dark, unit flat, no bad pixels) is built by
flight.payload.calibration_io.build_identity_calibration.

Contains:
  - MosaicCalibration: per-channel dark/flat/bad-pixel artifacts, shape (3, H, W).
  - correct_bad_pixels: replace masked pixels with their one-pixel neighbor mean.
  - calibrate_mosaic: bad-pixel repair then (repaired - dark) / flat on each channel.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass

# third-party
import numpy as np

# internal
from flight.libs.types import Err, FaultCode, Ok, Result


@dataclass(frozen=True, slots=True)
class MosaicCalibration:
    """Per-channel calibration for stacked prism planes.

    Loaded once at startup from checksummed artifacts (flight) or built as identity
    (SIL). Applied after stack_channels, per channel.

    Attributes:
        dark_frame: np.ndarray[float32, (3, H, W)] per-pixel dark signal in DN.
        flat_field: np.ndarray[float32, (3, H, W)] normalized response map, values ~1.0.
            A zero element causes a non-finite output; calibrate_mosaic catches this.
        bad_pixel_mask: np.ndarray[bool, (3, H, W)] True where the pixel is unusable.
            Bad pixels are repaired by correct_bad_pixels before dark/flat correction.

    Notes:
        All three arrays must share the same (3, H, W) shape, matching the stacked
        sensor frame. Mismatch is caught at calibrate_mosaic call time.
    """

    dark_frame: np.ndarray  # (3, H, W) float32
    flat_field: np.ndarray  # (3, H, W) float32, values ~1.0
    bad_pixel_mask: np.ndarray  # (3, H, W) bool


def correct_bad_pixels(planes: np.ndarray, bad_pixel_mask: np.ndarray) -> np.ndarray:
    """Replace bad pixels with the mean of their four one-pixel neighbors.

    Neighbors stay inside the same channel. Edge pixels use reflected padding
    (mode="reflect") so boundary bad pixels are also corrected.

    Single-pass: a bad neighbor contributes its raw value. This is acceptable for
    isolated defects; clustered defects should be excluded at sensor characterization
    time (not flagged at runtime).

    Args:
        planes: np.ndarray[float32, (3, H, W)] stacked channels (any numeric dtype
            accepted; output is float32).
        bad_pixel_mask: np.ndarray[bool, (3, H, W)] True marks pixels to replace.

    Returns:
        np.ndarray[float32, (3, H, W)] with bad pixels replaced; good pixels
        are returned unchanged (values cast to float32).
    """
    padded = np.pad(planes, ((0, 0), (1, 1), (1, 1)), mode="reflect")
    neighbors = (
        padded[:, :-2, 1:-1] + padded[:, 2:, 1:-1] + padded[:, 1:-1, :-2] + padded[:, 1:-1, 2:]
    ) / 4.0
    return np.where(bad_pixel_mask, neighbors, planes).astype(np.float32)


def calibrate_mosaic(
    planes: np.ndarray,
    cal: MosaicCalibration,
) -> Result[np.ndarray, FaultCode]:
    """Bad-pixel repair then (repaired - dark) / flat on stacked channels.

    Applies the calibration order: bad pixels are interpolated first (so they do
    not pollute the dark/flat statistics), then dark-frame subtraction, then
    flat-field correction. All operations are elementwise on ``(3, H, W)``.

    Args:
        planes: np.ndarray[float32, (3, H, W)] stacked channels. Shape must match
            cal.dark_frame.shape.
        cal: MosaicCalibration with dark_frame, flat_field, and bad_pixel_mask
            all of shape (3, H, W).

    Returns:
        Ok(np.ndarray[float32, (3, H, W)]) -- calibrated DN values, all finite.
        Err(FaultCode.FRAME_MALFORMED) -- planes.shape != cal.dark_frame.shape.
        Err(FaultCode.INFERENCE_NAN) -- any output pixel is non-finite. This covers a
            division by a zero flat-field element as well as a non-finite value already
            present in the input.

    Notes:
        Clipping of calibrated values to [0, full_scale] is NOT performed here; that
        is the responsibility of normalize_dn, which clips before scaling to [0, 1].
    """
    if planes.shape != cal.dark_frame.shape:
        return Err(FaultCode.FRAME_MALFORMED)
    repaired = correct_bad_pixels(planes, cal.bad_pixel_mask)
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected = (repaired - cal.dark_frame) / cal.flat_field
    if not np.isfinite(corrected).all():
        return Err(FaultCode.INFERENCE_NAN)
    return Ok(corrected)
