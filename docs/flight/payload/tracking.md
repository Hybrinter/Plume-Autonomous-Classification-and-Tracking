# flight.payload.tracking

**Source:** `packages/flight/src/flight/payload/tracking`
**Kind:** package

## Purpose

The tracking package holds pure target-state estimation and blob association helpers.
EMA smoothing feeds the Kalman filter; IoU matching assigns persistent blob IDs across
frames.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`filter`](tracking/filter.md) | pure module | EMA centroid smoothing in boresight-error space |
| [`kalman`](tracking/kalman.md) | pure module | Per-axis 4-state Kalman predict and updates |
| [`rewind`](tracking/rewind.md) | pure module | Vision-latency snapshot ring |
| [`tracker`](tracking/tracker.md) | pure module | IoU blob matching and persistence counting |

## Package interface

Re-exports: `AxisKalmanState`, `DualKalmanState`, `EmaFilterState`, `EstimatorRing`,
`EstimatorSnapshot`, `KalmanFilter`, `apply_vision`, `compute_iou`, `ema_update`,
`empty_ring`, `match_blobs`, `predict`, `push_snapshot`, `update_enc`, `update_vis`.

## Interactions

None. `PayloadController.ingest_vision` and `step_outer` call these functions.
Outputs update `ControlState` fields consumed by the arbiter and LQR.

## Constraints

All functions are pure with threaded immutable state. The EMA and Kalman operate in
boresight-error degrees after pointing conversion in the controller.

## Related documents

- [`flight.payload.control`](../control.md)
- [`flight.payload.gimbal.pointing`](../gimbal/pointing.md)
