# flight.payload.preprocess.quality

**Source:** `packages/flight/src/flight/payload/preprocess/quality.py`
**Kind:** pure module

## Purpose

This module computes per-frame quality flags on calibrated, normalized band data. Flags attach to
the local processed frame message.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SATURATION_PIXEL_LEVEL` | constant | Normalized DN level 0.95 for saturation tests |
| `compute_quality_flags` | function | Returns a frozenset of `FrameUsabilityTag` |

## Inputs and outputs

`compute_quality_flags(bands, exposure_us, slew_rate_deg_per_s, ifov_deg_per_px, utc_timestamp,
cfg)` returns `frozenset[FrameUsabilityTag]`. An empty set means no flags.

## Behavior

1. **INCOMPLETE_METADATA** when exposure is nonpositive or timestamp is empty.
2. **SATURATED** when any band has more than `saturation_fraction_threshold` of pixels above
   0.95.
3. **MOTION_SMEAR** when `slew_rate_deg_per_s * (exposure_us * 1e-6) / ifov_deg_per_px` exceeds
   `max_motion_smear_px`.
4. **CLOUD_CONTAMINATED** when mean NIR over mean RED exceeds `nir_red_ratio_threshold`. Band
   index 2 is RED; index 3 is NIR after `select_bands`.
5. **SUNGLINT** when mean NIR exceeds `sunglint_nir_mean_threshold`.

## Errors and faults

None.

## Messages

None. Flags are stored on `ProcessedFrameMsg.quality_flags` inside the app.

## Configuration

Reads `PreprocessingConfig` thresholds and `SensorConfig.ifov_deg_per_px`.

## Constraints

Pure module. Slew rate 0.0 never raises MOTION_SMEAR. Requires at least four bands for cloud and
sunglint checks.

## Related documents

- [`flight.payload.preprocess`](preprocess.md)
- [`flight.payload.app`](app.md)
