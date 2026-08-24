# flight.payload.preprocess.radiometric

**Source:** `packages/flight/src/flight/payload/preprocess/radiometric.py`
**Kind:** pure module

## Purpose

This module calibrates the raw mosaic plane before CFA separation. It repairs bad pixels, subtracts
dark signal, and divides by flat field.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `MosaicCalibration` | class | Frozen dark, flat, and bad-pixel mask arrays |
| `correct_bad_pixels` | function | Replaces masked pixels with same-band neighbor mean |
| `calibrate_mosaic` | function | Repair then `(repaired - dark) / flat` |

## Inputs and outputs

`correct_bad_pixels(mosaic, bad_pixel_mask)` returns a float32 `(H, W)` array.

`calibrate_mosaic(mosaic, cal)` returns `Result[np.ndarray, FaultCode]`.

## Behavior

1. `correct_bad_pixels` pads with reflect mode and averages four ±2 neighbors in the same CFA
   cell.
2. `calibrate_mosaic` checks shape match, runs bad-pixel repair, then elementwise dark subtract
   and flat divide.
3. Non-finite output returns `Err(INFERENCE_NAN)`.

Clipping to full scale happens in `normalize_dn`, not here.

## Errors and faults

| Result | Trigger |
| --- | --- |
| `Err(FRAME_MALFORMED)` | Mosaic shape differs from calibration shape |
| `Err(INFERENCE_NAN)` | Any non-finite calibrated pixel |

## Messages

None.

## Configuration

Artifact shape matches `SensorConfig.height_px` and `width_px`. Values load via
`calibration_io`.

## Constraints

Pure module. Calibration applies to the raw `(H, W)` mosaic, not band planes.

## Related documents

- [`flight.payload.preprocess`](preprocess.md)
- [`flight.payload.calibration_io`](calibration_io.md)
