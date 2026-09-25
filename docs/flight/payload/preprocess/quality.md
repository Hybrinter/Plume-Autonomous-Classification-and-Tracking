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
| `SmearRateSource` | enum | `MEASURED`, `ENCODER`, or `COMMANDED` provenance of the smear rate |
| `compute_quality_flags` | function | Returns a frozenset of raised usability tags |

## Inputs and outputs

`compute_quality_flags(bands, exposure_us, slew_rate_deg_per_s, ifov_band_deg_per_px,
utc_timestamp, cfg, omega_scene_el_deg_per_s=0.0)` returns
`frozenset[FrameUsabilityTag]`. An empty set means a clean frame.

`SmearRateSource` names the elevation-rate source the payload app selected for
MOTION_SMEAR. A rate of `0.0` is valid for every source. Unknown measured rate
plus a failed encoder bracket uses the commanded rate and labels it `COMMANDED`.

## Behavior

1. Raise `INCOMPLETE_METADATA` when exposure is nonpositive or the timestamp is empty.
2. Raise `SATURATED` when any band exceeds `saturation_fraction_threshold` of pixels
   above `SATURATION_PIXEL_LEVEL`.
3. Compute elevation-relative smear length as
   `abs(slew_rate_deg_per_s - omega_scene_el_deg_per_s) * exposure_s / ifov`.
   Raise `MOTION_SMEAR` when it exceeds `max_motion_smear_px`. Azimuth motion does not
   contribute. Matching rates, including both zero, do not flag.

## Errors and faults

None.

## Messages

None. Flags are carried on the in-process processed frame; they are not bus messages.

## Configuration

Reads `PreprocessingConfig`: `saturation_fraction_threshold`, `max_motion_smear_px`.
Also uses `SensorOpticsConfig.ifov_band_deg_per_px` and frame metadata from the
camera buffer.

## Constraints

Quality evaluation runs on the selected channels before the batch axis is added.
A zero gimbal rate is a stationary
measurement. The payload app supplies a measured, encoder, or commanded elevation
rate and a `SmearRateSource` label.

## Related documents

- [`flight.payload.preprocess`](preprocess.md)
- [`flight.payload.app`](../app.md)
