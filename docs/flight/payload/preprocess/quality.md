# flight.payload.preprocess.quality

**Source:** `packages/flight/src/flight/payload/preprocess/quality.py`
**Kind:** pure module

## Purpose

This module computes per-frame quality flags for a calibrated, normalized multispectral
tensor. Flags attach to the processed frame record and gate downstream inference and
dataset classification.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SATURATION_PIXEL_LEVEL` | constant | Normalized DN level (0.95) for saturation counting |
| `compute_quality_flags` | function | Returns a frozenset of raised usability tags |

## Inputs and outputs

`compute_quality_flags(bands, exposure_us, slew_rate_deg_per_s, ifov_band_deg_per_px,
utc_timestamp, cfg)` returns `frozenset[FrameUsabilityTag]`. An empty set means a clean
frame.

## Behavior

1. Raise `INCOMPLETE_METADATA` when exposure is nonpositive or the timestamp is empty.
2. Raise `SATURATED` when any band exceeds `saturation_fraction_threshold` of pixels
   above `SATURATION_PIXEL_LEVEL`.
3. Compute smear length as `slew_rate * exposure_s / ifov`. Raise `MOTION_SMEAR` when
   it exceeds `max_motion_smear_px`.
4. Raise `CLOUD_CONTAMINATED` when the NIR-to-Red mean ratio exceeds
   `nir_red_ratio_threshold` (bands at indices 2 and 3 after select).
5. Raise `SUNGLINT` when mean NIR exceeds `sunglint_nir_mean_threshold`.

## Errors and faults

None.

## Messages

None. Flags are carried on the in-process processed frame; they are not bus messages.

## Configuration

Reads `PreprocessingConfig`: `saturation_fraction_threshold`, `max_motion_smear_px`,
`nir_red_ratio_threshold`, `sunglint_nir_mean_threshold`. Also uses
`SensorConfig.ifov_band_deg_per_px` and frame metadata from the raw mosaic.

## Constraints

Quality evaluation runs on the full band plane. A slew
rate of 0.0 disables motion smear flagging when the rate is unknown.

## Related documents

- [`flight.payload.preprocess`](preprocess.md)
- [`flight.payload.app`](../app.md)
