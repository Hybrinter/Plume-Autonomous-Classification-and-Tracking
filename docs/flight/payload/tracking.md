# flight.payload.tracking

**Source:** `packages/flight/src/flight/payload/tracking`
**Kind:** package

## Purpose

The tracking package holds the two-state residual Kalman filter and blob association
helpers. IoU matching assigns persistent blob IDs across frames.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`residual`](tracking/residual.md) | pure module | Elevation-error / residual-rate filter |
| [`tracker`](tracking/tracker.md) | pure module | IoU blob matching and persistence counting |

## Package interface

Re-exports: `ResidualFilter`, `ResidualSnapshot`, `ResidualState`, `compute_iou`,
`match_blobs`, `predict`, `push_snapshot`, `rewind_update`, `update`.

## Interactions

`PayloadController` calls these functions directly. Outputs update `ControlState`
fields consumed by the outer law.

## Constraints

All functions are pure with threaded immutable state. Residual state is SI radians
and rad/s.

## Related documents

- [`flight.payload.control`](../control.md)
- [`flight.payload.gimbal.pointing`](../gimbal/pointing.md)
