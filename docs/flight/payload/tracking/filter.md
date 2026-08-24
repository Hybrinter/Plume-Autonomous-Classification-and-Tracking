# flight.payload.tracking.filter

**Source:** `packages/flight/src/flight/payload/tracking/filter.py`
**Kind:** pure module

## Purpose

The EMA filter smooths boresight-error centroid coordinates across frames. It reduces jitter
before Kalman update and LQR refinement.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `EmaFilterState` | class | Smoothed centroid and initialization flag |
| `ema_update` | function | One EMA step on (x, y) |

## Inputs and outputs

`ema_update(state, new_centroid, alpha)` returns a new `EmaFilterState`.

## Behavior

1. When `state.initialized` is false, return the raw centroid with `initialized=True`.
2. Otherwise blend: `smoothed = alpha * new + (1 - alpha) * previous` per axis.
3. When the control core loses a match, it resets to uninitialized with zero centroid.

## Errors and faults

None.

## Messages

None.

## Configuration

Reads `ControllerConfig.ema_alpha`.

## Constraints

Pure module. The control core feeds degree-space error from `boresight_error_deg`, not raw tensor
pixels.

## Related documents

- [`flight.payload.tracking`](tracking.md)
- [`flight.payload.control`](control.md)
