# flight.payload.preprocess.quality

**Source:** `packages/flight/src/flight/payload/preprocess/quality.py`
**Kind:** pure module

## Purpose

This module computes per-frame quality flags for a calibrated, normalized multispectral
tensor, the numeric metrics behind them, and the keep/drop verdict
(TRAINING / TRACKING / INVALID) that classifies the frame for the model interface and
the training dataset.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SATURATION_PIXEL_LEVEL` | constant | Saturation level as a fraction of full scale (0.95) |
| `DEFAULT_BAND_NAMES` | constant | The 4-band order assumed when `band_names` is not given |
| `CloudTest` | dataclass | Thresholds for the bright-and-spectrally-flat cloud test |
| `QualityMetrics` | dataclass | Numeric metrics behind the flags, for telemetry and tests |
| `UsabilityPolicy` | dataclass | Which flags make a frame INVALID and which TRACKING-only |
| `DEFAULT_USABILITY_POLICY` | constant | Recommended invalid/tracking-only flag split |
| `saturated_fraction_normalized` | function | Per-band saturated fraction on normalized planes |
| `saturated_fraction_raw` | function | Per-CFA-cell saturated fraction on the raw mosaic |
| `predicted_smear_px` | function | Gimbal plus platform smear length in band-plane pixels |
| `cloud_fraction` | function | Fraction of bright, spectrally flat pixels |
| `compute_quality_metrics` | function | Evaluate every metric; `Err(FRAME_MALFORMED)` on bad shapes |
| `flags_from_metrics` | function | Raise flags from metrics and thresholds |
| `compute_quality_flags` | function | Metrics to flags in one call (legacy entry point) |
| `decide_usability` | function | Map raised flags to TRAINING, TRACKING, or INVALID |

## Inputs and outputs

`compute_quality_flags(bands, exposure_us, slew_rate_deg_per_s, ifov_band_deg_per_px,
utc_timestamp, cfg, band_names, gain_db, raw_mosaic, adc_max_dn, cfa_phase,
platform_rate_deg_per_s, cloud_test, saturation_level)` returns
`frozenset[FrameUsabilityTag]`. An empty set means a clean frame;
`frozenset({INVALID})` means the input was malformed.

`compute_quality_metrics` returns `Result[QualityMetrics, FaultCode]`;
`flags_from_metrics(metrics, cfg, cloud_test)` returns the frozenset;
`decide_usability(flags, policy)` returns one `FrameUsabilityTag`.

## Behavior

1. Raise `INCOMPLETE_METADATA` when exposure is nonpositive, the timestamp is empty, or
   a supplied gain is negative.
2. Raise `SATURATED` when any band exceeds `saturation_fraction_threshold` of pixels
   above the saturation level. With `raw_mosaic` and `adc_max_dn` supplied the fraction
   is measured per CFA cell class on the raw mosaic; otherwise per normalized band.
3. Compute smear length as `(|slew_rate| + platform_rate) * exposure_s / ifov`. Raise
   `MOTION_SMEAR` when it exceeds `max_motion_smear_px`.
4. Raise `CLOUD_CONTAMINATED` from the `cloud_fraction` bright-and-flat test when a
   `CloudTest` is supplied, else from the NIR-to-RED mean ratio against
   `nir_red_ratio_threshold`.
5. Raise `SUNGLINT` when mean NIR exceeds `sunglint_nir_mean_threshold`.
6. `decide_usability` returns INVALID when the flags contain INVALID or any of
   `policy.invalid_flags`, TRACKING when any of `policy.tracking_only_flags`, else
   TRAINING.

## Errors and faults

`compute_quality_metrics` and `saturated_fraction_raw` return
`Err(FaultCode.FRAME_MALFORMED)` on wrong rank, channel-count, or an unseparable raw
mosaic.

## Messages

None. Flags are carried on the in-process processed frame; they are not bus messages.

## Configuration

Reads `PreprocessingConfig`: `saturation_fraction_threshold`, `max_motion_smear_px`,
`nir_red_ratio_threshold`, `sunglint_nir_mean_threshold`. Also uses
`SensorConfig.ifov_band_deg_per_px` and frame metadata from the raw mosaic.

## Constraints

Quality evaluation runs on the full band plane. Band identity resolves by name through
`band_names`; when RED or NIR is absent the spectral metrics are NaN and their flags
are not raised. A slew rate of 0.0 with zero platform rate disables motion smear
flagging when the rate is unknown.

## Related documents

- [`flight.payload.preprocess`](preprocess.md)
- [`flight.payload.app`](../app.md)
