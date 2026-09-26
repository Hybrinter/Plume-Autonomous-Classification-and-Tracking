"""Per-channel radiometric calibration for the JAI prism RGB frame.

Satisfies: REQ-AIML-PREP-002

Each color is its own CMOS. Dark, flat, and bad pixels are (3, H, W) arrays in
``BAND_ORDER`` (BLUE, GREEN, RED). There is no mosaic and no CFA.

Pipeline (calibrate_mosaic):
    corrected = (planes - dark_frame) / flat_field
    repaired  = correct_bad_pixels(corrected, bad_pixel_mask)

Dark and flat run first. Bad pixels are repaired from already-corrected
neighbors on the same channel. A neighbor step of 1 stays on that channel.

A non-finite result is Err(INFERENCE_NAN). A shape mismatch is
Err(FRAME_MALFORMED). A recorded dark exposure or gain that does not match
the frame is Err(CALIBRATION_INVALID).

Contains:
  - MosaicCalibration: per-channel dark, flat, bad-pixel mask, and the exposure
    and gain the dark frames were taken at.
  - dark_matches_frame: True when supplied exposure and gain match the dark.
  - correct_bad_pixels: replace masked pixels with good neighbor means.
  - calibrate_mosaic: dark/flat, then bad-pixel repair, on (3, H, W).
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass

# third-party
import numpy as np

# internal
from flight.libs.types import BAND_ORDER, Err, FaultCode, Ok, Result

_AXIAL: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAGONAL: tuple[tuple[int, int], ...] = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_PAD: int = 1
_BandTriple = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class MosaicCalibration:
    """Per-channel dark, flat, and bad-pixel maps in ``BAND_ORDER``.

    Attributes:
        dark_frame: np.ndarray[float32, (3, H, W)] dark signal in DN.
        flat_field: np.ndarray[float32, (3, H, W)] response, values near 1.
        bad_pixel_mask: np.ndarray[bool, (3, H, W)] True where the pixel is bad.
        dark_exposure_us: exposure of each dark plane, or None if unrecorded.
        dark_gain_db: gain of each dark plane, or None if unrecorded.
    """

    dark_frame: np.ndarray  # (3, H, W) float32
    flat_field: np.ndarray  # (3, H, W) float32
    bad_pixel_mask: np.ndarray  # (3, H, W) bool
    dark_exposure_us: _BandTriple | None = None
    dark_gain_db: _BandTriple | None = None


def _as_triple(value: float | _BandTriple | None) -> _BandTriple | None:
    """Broadcast a shared exposure or gain onto the three bands."""
    if value is None:
        return None
    if isinstance(value, tuple):
        return value
    return (value, value, value)


def dark_matches_frame(
    cal: MosaicCalibration,
    exposure_us: float | _BandTriple | None = None,
    gain_db: float | _BandTriple | None = None,
    exposure_tolerance_frac: float = 0.05,
    gain_tolerance_db: float = 0.5,
) -> bool:
    """Return True when the dark frames match the supplied exposure and gain.

    A field is checked only when both the frame and the calibration record it.
    A float applies to every band. A triple is ``BAND_ORDER``.

    Inputs:
        cal (MosaicCalibration): Dark exposure and gain to compare.
        exposure_us (float | tuple | None): Frame exposure in microseconds.
        gain_db (float | tuple | None): Frame analog gain in dB.
        exposure_tolerance_frac (float): Allowed fractional exposure error.
        gain_tolerance_db (float): Allowed gain error in dB.

    Outputs:
        bool: True when every recorded field is inside tolerance.
    """
    frame_exp = _as_triple(exposure_us)
    frame_gain = _as_triple(gain_db)
    if frame_exp is not None and cal.dark_exposure_us is not None:
        for got, expected in zip(frame_exp, cal.dark_exposure_us, strict=True):
            if abs(got - expected) > exposure_tolerance_frac * abs(expected):
                return False
    if frame_gain is not None and cal.dark_gain_db is not None:
        for got, expected in zip(frame_gain, cal.dark_gain_db, strict=True):
            if abs(got - expected) > gain_tolerance_db:
                return False
    return True


def _neighbour_mean(
    padded: np.ndarray,
    padded_good: np.ndarray,
    offsets: tuple[tuple[int, int], ...],
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum and count of good neighbors at ``offsets`` for one plane."""
    total = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    for d_row, d_col in offsets:
        r0, c0 = _PAD + d_row, _PAD + d_col
        vals = padded[r0 : r0 + height, c0 : c0 + width]
        good = padded_good[r0 : r0 + height, c0 : c0 + width]
        total += np.where(good > 0.0, vals, 0.0)
        count += good
    return total, count


def _repair_plane(plane: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace bad pixels on one channel with good axial, then diagonal, neighbors."""
    height, width = plane.shape
    padded = np.pad(plane, _PAD, mode="reflect")
    good_f = (~mask).astype(np.float32)
    padded_good = np.pad(good_f, _PAD, mode="reflect")
    axial_sum, axial_n = _neighbour_mean(padded, padded_good, _AXIAL, height, width)
    diag_sum, diag_n = _neighbour_mean(padded, padded_good, _DIAGONAL, height, width)
    use_diag = axial_n == 0.0
    total = np.where(use_diag, diag_sum, axial_sum)
    count = np.where(use_diag, diag_n, axial_n)
    repairable = count > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        replacement = np.where(repairable, total / count, plane)
    return np.where(mask, replacement, plane).astype(np.float32)


def correct_bad_pixels(planes: np.ndarray, bad_pixel_mask: np.ndarray) -> np.ndarray:
    """Replace bad pixels on each RGB channel independently.

    Inputs:
        planes (np.ndarray[float32, (3, H, W)]): Planes in ``BAND_ORDER``.
        bad_pixel_mask (np.ndarray[bool, (3, H, W)]): True marks pixels to replace.

    Outputs:
        np.ndarray[float32, (3, H, W)]: Planes with bad pixels replaced.
    """
    stacked = planes.astype(np.float32, copy=False)
    out = np.empty_like(stacked)
    for index in range(stacked.shape[0]):
        out[index] = _repair_plane(stacked[index], bad_pixel_mask[index])
    return out


def calibrate_mosaic(
    planes: np.ndarray,
    cal: MosaicCalibration,
    exposure_us: float | _BandTriple | None = None,
    gain_db: float | _BandTriple | None = None,
) -> Result[np.ndarray, FaultCode]:
    """Subtract dark, divide by flat, then repair bad pixels on each channel.

    Inputs:
        planes (np.ndarray): Raw (3, H, W) planes in ``BAND_ORDER``.
        cal (MosaicCalibration): Matching dark, flat, and bad-pixel maps.
        exposure_us (float | tuple | None): Frame exposure. See dark_matches_frame.
        gain_db (float | tuple | None): Frame gain. See dark_matches_frame.

    Outputs:
        Result[np.ndarray, FaultCode]:
            Ok((3, H, W) float32) when every value is finite.
            Err(FRAME_MALFORMED) on rank, band count, or shape mismatch.
            Err(CALIBRATION_INVALID) when exposure or gain does not match.
            Err(INFERENCE_NAN) when any output value is NaN or Inf.
    """
    if planes.ndim != 3 or planes.shape[0] != len(BAND_ORDER):
        return Err(FaultCode.FRAME_MALFORMED)
    if (
        planes.shape != cal.dark_frame.shape
        or planes.shape != cal.flat_field.shape
        or planes.shape != cal.bad_pixel_mask.shape
    ):
        return Err(FaultCode.FRAME_MALFORMED)
    if not dark_matches_frame(cal, exposure_us, gain_db):
        return Err(FaultCode.CALIBRATION_INVALID)
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected = (planes.astype(np.float32) - cal.dark_frame) / cal.flat_field
    if not np.isfinite(corrected).all():
        return Err(FaultCode.INFERENCE_NAN)
    return Ok(correct_bad_pixels(corrected, cal.bad_pixel_mask))
