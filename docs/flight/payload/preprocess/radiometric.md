# flight.payload.preprocess.radiometric

**Source:** `packages/flight/src/flight/payload/preprocess/radiometric.py`
**Kind:** pure module

## Purpose

This module applies radiometric calibration on the raw mosaic plane: it subtracts the
dark frame and divides by the flat field, then repairs bad pixels in the corrected
domain. It also carries the exposure and gain the dark frame was acquired at and can
check a frame's metadata against them.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `MosaicCalibration` | dataclass | Per-pixel dark, flat, bad-pixel mask, and dark acquisition metadata |
| `dark_matches_frame` | function | Per-field tolerance check of frame exposure/gain against the dark's |
| `correct_bad_pixels` | function | Replaces masked pixels with same-band neighbor mean |
| `calibrate_mosaic` | function | Dark subtraction, flat correction, then bad-pixel repair |

## Inputs and outputs

`calibrate_mosaic(mosaic, cal, exposure_us=None, gain_db=None)` returns
`Result[np.ndarray, FaultCode]` with a float32 `(H, W)` calibrated plane.

`dark_matches_frame(cal, exposure_us=None, gain_db=None, exposure_tolerance_frac=0.05,
gain_tolerance_db=0.5)` returns bool. Each field is checked only when the frame
supplies it AND the calibration records it (`None` means unrecorded on either side).

`correct_bad_pixels(mosaic, bad_pixel_mask)` returns a float32 `(H, W)` array.

## Behavior

1. `calibrate_mosaic` checks that the mosaic shape matches all three calibration
   artifacts.
2. It checks supplied frame exposure/gain against the dark frame's recorded
   acquisition values (`dark_matches_frame`).
3. It computes `(mosaic - dark) / flat` elementwise, then repairs bad pixels from
   good same-band neighbours only, at +/-2 offsets (axial ring first, diagonal ring
   as fallback), so a hot pixel's own dark/flat entries never influence its
   replacement.
4. It rejects the result when any value is non-finite.

## Errors and faults

| Result | Trigger |
| --- | --- |
| `Err(FRAME_MALFORMED)` | Mosaic shape differs from `dark_frame`, `flat_field`, or `bad_pixel_mask` shape |
| `Err(CALIBRATION_INVALID)` | A supplied exposure/gain is outside tolerance of the dark frame's recorded values |
| `Err(INFERENCE_NAN)` | Any output pixel is NaN or Inf after repair (includes zero flat field) |

## Messages

None.

## Configuration

None directly. Calibration artifacts come from `MosaicCalibration`, loaded at startup
from `SensorConfig` geometry via `calibration_io`. `dark_exposure_us`/`dark_gain_db`
are `None` unless the loader records the dark's acquisition point.

## Constraints

Calibration runs on the full mosaic plane before CFA separation. Dark and flat apply
first and repair runs last; repairing in the corrected domain keeps the replacement
physically consistent. Clipping to [0, full scale] happens in `normalize_dn`, not
here.

## Related documents

- [`flight.payload.preprocess`](preprocess.md)
- [`flight.payload.calibration_io`](../calibration_io.md)
