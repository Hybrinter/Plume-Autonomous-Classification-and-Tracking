# flight.payload.preprocess.radiometric

**Source:** `packages/flight/src/flight/payload/preprocess/radiometric.py`
**Kind:** pure module

## Purpose

This module applies radiometric calibration on a raw RGB stack. Each color is its
own CMOS. Dark, flat, and bad pixels are `(3, H, W)` arrays in `BAND_ORDER`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `MosaicCalibration` | dataclass | Per-channel dark, flat, bad-pixel mask, and dark exposure and gain |
| `dark_matches_frame` | function | True when recorded dark exposure and gain match the frame |
| `correct_bad_pixels` | function | Replaces masked pixels with same-channel neighbor means |
| `calibrate_mosaic` | function | Dark and flat, then bad-pixel repair |

## Inputs and outputs

`correct_bad_pixels(planes, bad_pixel_mask)` returns float32 `(3, H, W)`.

`calibrate_mosaic(planes, cal, exposure_us=None, gain_db=None)` returns
`Result[np.ndarray, FaultCode]` with a float32 `(3, H, W)` stack.

A float exposure or gain applies to every channel. A triple follows `BAND_ORDER`.

## Behavior

1. Reject a rank other than 3, a channel count other than `len(BAND_ORDER)`, or a
   shape that misses the calibration arrays.
2. Reject the frame when a recorded dark exposure or gain is outside tolerance.
3. Compute `(planes - dark) / flat` on each channel.
4. Reject the result when any value is non-finite.
5. Replace each bad pixel with the mean of good neighbors at +/-1. Axial neighbors
   are used first. Diagonal neighbors are used when no axial neighbor is good.

## Errors and faults

| Result | Trigger |
| --- | --- |
| `Err(FRAME_MALFORMED)` | Rank, channel count, or shape does not match the calibration |
| `Err(CALIBRATION_INVALID)` | Dark exposure or gain does not match the frame |
| `Err(INFERENCE_NAN)` | Any output pixel is NaN or Inf, including a zero flat field |

## Messages

None.

## Configuration

None directly. Calibration artifacts come from `MosaicCalibration`, loaded at startup
from `SensorConfig` geometry via `calibration_io`.

## Constraints

Dark and flat run before bad-pixel repair. A neighbor step of 1 stays on that
channel. Clipping to [0, 1] happens in `normalize_dn`, not here.

## Related documents

- [`flight.payload.preprocess`](../preprocess.md)
- [`flight.payload.calibration_io`](../calibration_io.md)
