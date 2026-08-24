# flight.payload.tracking

**Source:** `packages/flight/src/flight/payload/tracking/`
**Kind:** package

## Purpose

The tracking package holds pure target-state helpers: EMA smoothing, a constant-velocity Kalman
filter, and IoU blob association.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`filter`](tracking/filter.md) | module | EMA centroid smoothing in degree space |
| [`kalman`](tracking/kalman.md) | module | 2-axis constant-velocity Kalman filter |
| [`tracker`](tracking/tracker.md) | module | IoU matching and persistence counting |

## Package interface

Re-exports: `EmaFilterState`, `ema_update`, `KalmanFilter`, `KalmanState`, `predict`, `update`,
`compute_iou`, `match_blobs`.

## Interactions

None. The control core calls these functions inside `PayloadController.step`.

## Constraints

All modules are pure. Blob IDs and persistence are assigned in `match_blobs` before the arbiter
uses persistence thresholds.

## Related documents

- [`flight.payload`](payload.md)
- [`flight.payload.control`](control.md)
