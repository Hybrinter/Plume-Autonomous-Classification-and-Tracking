# flight.payload.tracking.rewind

**Source:** `packages/flight/src/flight/payload/tracking/rewind.py`
**Kind:** pure module

## Purpose

This module stores outer-loop Kalman snapshots and applies a delayed vision
measurement at shutter time. It then predicts forward to the current time.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `EstimatorSnapshot` | dataclass | Timestamped dual-axis state, rate command, and encoder |
| `EstimatorRing` | dataclass | Bounded snapshot buffer |
| `empty_ring` | function | Returns an empty ring |
| `push_snapshot` | function | Appends a snapshot and drops the oldest on overflow |
| `apply_vision` | function | Roll back, update once, replay to now |

## Inputs and outputs

`apply_vision` takes the filter, ring, vision `z` pair, shutter time, current time,
held rate commands, and the current dual state. It returns `DualKalmanState` or `None`
when the shutter is older than the ring.

## Behavior

1. Find the last snapshot with `t <= t_shutter`.
2. Apply `update_vis` once to each axis at that snapshot.
3. Replay `predict` and encoder updates through later snapshots.
4. Predict from the last snapshot to `now` with the current held rate.

## Errors and faults

None. A failed vision update returns `None`.

## Messages

None.

## Configuration

Ring capacity comes from `ControllerConfig.estimator_ring_len` at the caller.

## Constraints

Each vision sample is applied once. The module is pure.

## Related documents

- [`flight.payload.tracking.kalman`](kalman.md)
- [`flight.payload.tracking`](../tracking.md)
