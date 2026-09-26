# flight.payload.preprocess.radiometric

**Source:** `packages/flight/src/flight/payload/preprocess/radiometric.py`
**Kind:** pure module

## Purpose

This module applies radiometric calibration on stacked prism channels. It repairs bad
pixels, subtracts dark signal, and divides by the flat field. Each artifact is
`(3, H, W)`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `MosaicCalibration` | dataclass | Per-channel dark, flat, and bad-pixel mask, shape `(3, H, W)` |
| `correct_bad_pixels` | function | Replaces masked pixels with the one-pixel neighbor mean |
| `calibrate_mosaic` | function | Bad-pixel repair then dark subtraction and flat correction |

## Inputs and outputs

`correct_bad_pixels(planes, bad_pixel_mask)` returns a float32 `(3, H, W)` array.

`calibrate_mosaic(planes, cal)` returns `Result[np.ndarray, FaultCode]` with a float32
`(3, H, W)` calibrated stack.

## Behavior

1. `correct_bad_pixels` pads each channel by one pixel, averages the four neighbors,
   and replaces pixels where the mask is true.
2. `calibrate_mosaic` checks that the stack shape matches the calibration artifacts.
3. It runs bad-pixel repair, then computes `(repaired - dark) / flat` elementwise.
4. It rejects the result when any value is non-finite.

## Errors and faults

| Result | Trigger |
| --- | --- |
| `Err(FRAME_MALFORMED)` | Stack shape differs from calibration shape |
| `Err(INFERENCE_NAN)` | Any output pixel is NaN or Inf (includes zero flat field) |

## Messages

None.

## Configuration

None directly. Calibration artifacts come from `MosaicCalibration`, loaded at startup
from `SensorConfig` geometry via `calibration_io`.

## Constraints

Calibration runs on the stacked channels. Neighbors stay inside one channel.
Clipping to [0, full scale] happens in `normalize_dn`, not here.

## Related documents

- [`flight.payload.preprocess`](preprocess.md)
- [`flight.payload.calibration_io`](../calibration_io.md)
