# flight.payload.tracking.filter

**Source:** `packages/flight/src/flight/payload/tracking/filter.py`
**Kind:** pure module

## Purpose

This module applies an exponential moving average to blob centroid coordinates in
boresight-error degree space. It reduces per-frame inference jitter before Kalman
update and LQR control.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `EmaFilterState` | dataclass | Smoothed centroid and initialization flag |
| `ema_update` | function | Blends a new centroid into the EMA state |

## Inputs and outputs

`ema_update(state, new_centroid, alpha)` takes the prior `EmaFilterState`, a `(x, y)`
centroid tuple, and smoothing factor `alpha`. It returns a new `EmaFilterState`.

## Behavior

1. When `state.initialized` is false, return the raw centroid with `initialized=True`
   and no blending.
2. On later frames, compute
   `smoothed = alpha * new + (1 - alpha) * previous` per axis.
3. Return the updated state with `initialized=True`.

When no blob match exists, `PayloadController` resets to an uninitialized EMA.

## Errors and faults

None.

## Messages

None.

## Configuration

Uses `ControllerConfig.ema_alpha` (default 0.4).

## Constraints

Centroids arrive in boresight-error degrees from `boresight_error_deg`, not raw tensor
pixels. State is immutable; each update returns a new `EmaFilterState`.

## Related documents

- [`flight.payload.tracking`](../tracking.md)
- [`flight.payload.control`](../control.md)
