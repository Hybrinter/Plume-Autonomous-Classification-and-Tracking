"""
flight.payload.preprocess.radiometric -- Mosaic-plane radiometric calibration.

Satisfies: REQ-AIML-PREP-002

Calibration runs on the RAW (H, W) mosaic plane, BEFORE CFA separation. Dark signal
(bias + dark current) and flat-field response are properties of the IMX264 sensor and
of the 2x2 CFA overlay in front of it, so they are characterised and applied in mosaic
(sensor) space.

Pipeline (calibrate_mosaic):
    corrected = (mosaic - dark_frame) / flat_field
    repaired  = correct_bad_pixels(corrected, bad_pixel_mask)

    Dark and flat are applied FIRST and bad pixels are repaired LAST. A bad pixel is
    typically a hot pixel with a large dark_frame entry and an abnormal flat_field entry.
    Repairing it before dark/flat would replace its raw value with the neighbour mean
    and then subtract its own (large) dark and divide by its own (abnormal) flat, which
    over-corrects the repaired value. Repairing in the corrected domain uses only good,
    already-corrected neighbours, so the replacement is physically consistent.

    If the corrected output contains any NaN or Inf (e.g. a zero flat_field pixel or a
    non-finite input value), calibrate_mosaic() returns Err(FaultCode.INFERENCE_NAN). A
    shape mismatch between the mosaic and the calibration artifacts returns
    Err(FaultCode.FRAME_MALFORMED). When the frame's exposure/gain are supplied and the
    calibration records the exposure/gain the dark frame was taken at, a mismatch beyond
    tolerance returns Err(FaultCode.CALIBRATION_INVALID): a dark frame is only valid at
    the exposure and gain it was acquired with.

Calibration artifacts (dark, flat, bad_pixel_mask) for flight are loaded from
checksummed .npy files by flight.payload.calibration_io. The SIL/dev identity
calibration (zero dark, unit flat, no bad pixels) is built by
flight.payload.calibration_io.build_identity_calibration.

Contains:
  - MosaicCalibration: per-pixel dark/flat/bad-pixel artifacts for the raw mosaic plane,
    plus the exposure/gain the dark frame was acquired at (0.0 = unrecorded).
  - dark_matches_frame: True when the frame exposure/gain are within tolerance of the
    dark frame's acquisition exposure/gain (or the calibration does not record them).
  - correct_bad_pixels: replace masked pixels with the mean of their GOOD same-band
    (+/-2) neighbours; falls back to the diagonal +/-2 ring, then leaves the pixel.
  - calibrate_mosaic: (mosaic - dark) / flat, then bad-pixel repair, on the mosaic plane.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass

# third-party
import numpy as np

# internal
from flight.libs.types import Err, FaultCode, Ok, Result

# Same-band neighbour offsets on a 2x2 CFA: a step of 2 along either axis lands on the
# same cell class, so these never mix spectral bands. Axial ring first, diagonal ring as
# fallback when every axial neighbour is itself bad.
_AXIAL_OFFSETS: tuple[tuple[int, int], ...] = ((-2, 0), (2, 0), (0, -2), (0, 2))
_DIAGONAL_OFFSETS: tuple[tuple[int, int], ...] = ((-2, -2), (-2, 2), (2, -2), (2, 2))
_PAD: int = 2


@dataclass(frozen=True, slots=True)
class MosaicCalibration:
    """Per-pixel calibration for the RAW mosaic plane (pre-demosaic).

    Loaded once at startup from checksummed artifacts (flight) or built as identity
    (SIL). Applied before CFA separation, where the physics lives: dark signal and
    flat-field response variation are sensor and CFA properties characterised in mosaic
    space.

    Attributes:
        dark_frame: np.ndarray[float32, (H, W)] per-pixel dark signal in DN (bias plus
            dark current integrated over dark_exposure_us).
        flat_field: np.ndarray[float32, (H, W)] normalized response map, values ~1.0.
            Because each 2x2 CFA cell class transmits a different passband, the flat
            must be normalized to mean 1.0 PER CELL CLASS (four stride-2 sub-grids), not
            globally; a globally normalized flat rescales the band ratios the quality
            heuristics depend on. Per-class normalization is the artifact producer's
            responsibility. A zero element causes a non-finite output; calibrate_mosaic
            catches this.
        bad_pixel_mask: np.ndarray[bool, (H, W)] True where the pixel is unusable.
            Bad pixels are repaired by correct_bad_pixels after dark/flat correction.
        dark_exposure_us: float exposure the dark frame was acquired at, microseconds.
            0.0 means unrecorded; the exposure match check is then skipped.
        dark_gain_db: float analog gain the dark frame was acquired at, dB. Only
            checked when dark_exposure_us is recorded (> 0).

    Notes:
        All three arrays must share the same (H, W) shape, matching the sensor mosaic
        dimensions from SensorConfig. Mismatch is caught at calibrate_mosaic call time.
    """

    dark_frame: np.ndarray  # (H, W) float32
    flat_field: np.ndarray  # (H, W) float32, values ~1.0, mean 1.0 per CFA cell class
    bad_pixel_mask: np.ndarray  # (H, W) bool
    dark_exposure_us: float = 0.0
    dark_gain_db: float = 0.0


def dark_matches_frame(
    cal: MosaicCalibration,
    exposure_us: float,
    gain_db: float,
    exposure_tolerance_frac: float = 0.05,
    gain_tolerance_db: float = 0.5,
) -> bool:
    """Return True when the dark frame is valid for a frame at exposure_us / gain_db.

    Dark signal is bias + dark_current * t_exp, and both terms scale with analog gain,
    so a dark frame is only a correct subtrahend at (or near) the exposure and gain it
    was acquired with. Without a separate bias frame the two terms cannot be scaled
    independently, so this module checks the match instead of rescaling.

    Inputs:
        cal (MosaicCalibration): Calibration whose dark_exposure_us / dark_gain_db are
            compared. If cal.dark_exposure_us <= 0.0 the acquisition point is
            unrecorded and the check passes unconditionally.
        exposure_us (float): Frame exposure, microseconds.
        gain_db (float): Frame analog gain, dB.
        exposure_tolerance_frac (float): Allowed |exposure - dark_exposure| as a
            fraction of dark_exposure.
        gain_tolerance_db (float): Allowed |gain - dark_gain| in dB.

    Outputs:
        bool: True when within tolerance (or unrecorded), False otherwise.
    """
    if cal.dark_exposure_us <= 0.0:
        return True
    exposure_ok = (
        abs(exposure_us - cal.dark_exposure_us) <= exposure_tolerance_frac * cal.dark_exposure_us
    )
    gain_ok = abs(gain_db - cal.dark_gain_db) <= gain_tolerance_db
    return exposure_ok and gain_ok


def _neighbour_mean(
    padded: np.ndarray,
    padded_good: np.ndarray,
    offsets: tuple[tuple[int, int], ...],
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum and count of GOOD neighbours at the given offsets for every pixel.

    Inputs:
        padded (np.ndarray[float32, (H+4, W+4)]): Reflect-padded plane.
        padded_good (np.ndarray[float32, (H+4, W+4)]): Reflect-padded 1.0/0.0 good mask.
        offsets (tuple[tuple[int, int], ...]): (row, col) neighbour offsets, |offset| <= 2.
        h (int): Plane height.
        w (int): Plane width.

    Outputs:
        tuple[np.ndarray, np.ndarray]: (weighted_sum, count), each
            np.ndarray[float32, (H, W)]; count is the number of good neighbours.
    """
    total = np.zeros((h, w), dtype=np.float32)  # np.ndarray[float32, (H, W)]
    count = np.zeros((h, w), dtype=np.float32)  # np.ndarray[float32, (H, W)]
    for dr, dc in offsets:
        r0, c0 = _PAD + dr, _PAD + dc
        vals = padded[r0 : r0 + h, c0 : c0 + w]  # np.ndarray[float32, (H, W)]
        good = padded_good[r0 : r0 + h, c0 : c0 + w]  # np.ndarray[float32, (H, W)]
        # np.where (not vals * good) so a non-finite value at a bad neighbour cannot
        # poison the sum: NaN * 0.0 is NaN.
        total += np.where(good > 0.0, vals, 0.0)
        count += good
    return total, count


def correct_bad_pixels(mosaic: np.ndarray, bad_pixel_mask: np.ndarray) -> np.ndarray:
    """Replace bad pixels with the mean of their GOOD same-band (+/-2) neighbours.

    Offsets of +/-2 along each axis stay inside the same 2x2 CFA cell class, so the
    replacement uses same-band data and does not mix spectral information. Edge pixels
    use reflected padding (mode="reflect", which maps index -1 -> 1 and -2 -> 2 and
    therefore preserves CFA parity) so boundary bad pixels are also corrected.

    Neighbours that are themselves masked are EXCLUDED from the mean, so a defect
    cluster does not contaminate the repair. Fallback order per bad pixel:
        1. mean of the good axial neighbours (+/-2 rows or +/-2 columns);
        2. if none are good, mean of the good diagonal neighbours (+/-2, +/-2);
        3. if none are good either (a fully bad 5x5 same-band neighbourhood), the
           pixel keeps its input value. Such clusters should be excluded at sensor
           characterization time.

    Inputs:
        mosaic (np.ndarray[float32, (H, W)]): Plane to repair (any numeric dtype
            accepted; output is float32). Normally the dark/flat-corrected mosaic.
        bad_pixel_mask (np.ndarray[bool, (H, W)]): True marks pixels to replace.

    Outputs:
        np.ndarray[float32, (H, W)]: Plane with bad pixels replaced; good pixels are
            returned unchanged (values cast to float32).
    """
    plane = mosaic.astype(np.float32, copy=False)  # np.ndarray[float32, (H, W)]
    h, w = plane.shape
    padded = np.pad(plane, _PAD, mode="reflect")  # np.ndarray[float32, (H+4, W+4)]
    good_f = (~bad_pixel_mask).astype(np.float32)  # np.ndarray[float32, (H, W)]
    padded_good = np.pad(good_f, _PAD, mode="reflect")  # np.ndarray[float32, (H+4, W+4)]

    axial_sum, axial_n = _neighbour_mean(padded, padded_good, _AXIAL_OFFSETS, h, w)
    diag_sum, diag_n = _neighbour_mean(padded, padded_good, _DIAGONAL_OFFSETS, h, w)

    use_diag = axial_n == 0.0
    total = np.where(use_diag, diag_sum, axial_sum)  # np.ndarray[float32, (H, W)]
    count = np.where(use_diag, diag_n, axial_n)  # np.ndarray[float32, (H, W)]
    repairable = count > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        replacement = np.where(repairable, total / count, plane)  # np.ndarray[float32, (H, W)]
    return np.where(bad_pixel_mask, replacement, plane).astype(np.float32)


def calibrate_mosaic(
    mosaic: np.ndarray,
    cal: MosaicCalibration,
    exposure_us: float | None = None,
    gain_db: float | None = None,
) -> Result[np.ndarray, FaultCode]:
    """(mosaic - dark) / flat, then bad-pixel repair, on the raw mosaic plane.

    Dark-frame subtraction and flat-field division are elementwise on the full (H, W)
    mosaic plane (before CFA separation). Bad pixels are then repaired in the corrected
    domain from good neighbours only, so a hot pixel's own dark/flat entries never
    influence its replacement value.

    Inputs:
        mosaic (np.ndarray[float32, (H, W)]): Raw mosaic plane. Shape must match
            cal.dark_frame.shape.
        cal (MosaicCalibration): dark_frame, flat_field, bad_pixel_mask of shape (H, W).
        exposure_us (float | None): Frame exposure, microseconds. When given together
            with gain_db and the calibration records its dark acquisition point, the
            dark frame is checked for validity at this exposure/gain.
        gain_db (float | None): Frame analog gain, dB. See exposure_us.

    Outputs:
        Result[np.ndarray, FaultCode]:
            Ok(np.ndarray[float32, (H, W)]) -- calibrated DN values, all finite.
            Err(FaultCode.FRAME_MALFORMED) -- mosaic.shape != cal.dark_frame.shape.
            Err(FaultCode.CALIBRATION_INVALID) -- exposure_us/gain_db supplied and the
                dark frame was acquired at a different exposure/gain (see
                dark_matches_frame).
            Err(FaultCode.INFERENCE_NAN) -- any output pixel is non-finite AFTER repair.
                This covers a division by a zero flat-field element at an unmasked
                pixel as well as a non-finite value already present in the input
                mosaic. A non-finite value at a MASKED pixel is repaired from its good
                neighbours and does not fault.

    Notes:
        Clipping of calibrated values to [0, full_scale] is NOT performed here; that is
        the responsibility of normalize_dn, which clips before scaling to [0, 1].
        Artifact validity (finite, positive flat; per-cell-class flat mean of 1.0) is
        the artifact producer's and loader's responsibility; this function only guards
        against a non-finite result.
    """
    if mosaic.shape != cal.dark_frame.shape:
        return Err(FaultCode.FRAME_MALFORMED)
    if exposure_us is not None and gain_db is not None:
        if not dark_matches_frame(cal, exposure_us, gain_db):
            return Err(FaultCode.CALIBRATION_INVALID)
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected = (mosaic - cal.dark_frame) / cal.flat_field  # np.ndarray[float32, (H, W)]
    repaired = correct_bad_pixels(corrected, cal.bad_pixel_mask)
    if not np.isfinite(repaired).all():
        return Err(FaultCode.INFERENCE_NAN)
    return Ok(repaired)
