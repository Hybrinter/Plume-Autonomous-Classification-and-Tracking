# flight.payload.tracking

**Source:** `packages/flight/src/flight/payload/tracking`

**Kind:** package

## Purpose

The tracking package holds the two-state residual estimator and blob association
helpers. IoU matching assigns persistent blob IDs across frames.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`residual`](tracking/residual.md) | pure module | Timestamped residual estimator and event replay |
| [`tracker`](tracking/tracker.md) | pure module | IoU blob matching and persistence counting |

## Package interface

The package re-exports `ResidualFilter`, `ResidualState`, `ResidualCheckpoint`,
`ResidualHistory`, `EncoderSample`, `NominalRateSample`,
`PredictorReferenceChange`, `VisionObservation`, `ObservationDisposition`,
`EventDisposition`, `EstimateResult`, `propagate_displacement`, `submit_event`,
`estimate_at`, `update`, `predict`, `ResidualSnapshot`, `rewind_update`,
`push_snapshot`, `compute_iou`, and `match_blobs`.

`ResidualSnapshot` and its helper functions are compatibility symbols for the
detailed-plant transition. The production outer path uses timestamped event
replay.

## Interactions

`PayloadController` submits encoder, nominal-rate, reference-change, and vision
events to `ResidualHistory`. It requests the estimate at the current encoder
sample time. The controller uses the resulting residual state and covariance in
the outer rate law.

## Constraints

All tracking functions are pure with immutable threaded state. Residual state
uses SI radians and radians per second. The package does not read a clock or
access a HAL driver.

## Related documents

- [`flight.payload.control`](../control.md)
- [`flight.payload.gimbal.pointing`](../gimbal/pointing.md)
