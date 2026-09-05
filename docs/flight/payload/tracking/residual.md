# flight.payload.tracking.residual

**Source:** `packages/flight/src/flight/payload/tracking/residual.py`
**Kind:** pure module

## Purpose

This module implements a two-state residual Kalman filter on elevation error `e` and
residual rate `omega_t_res`. Encoder rate `y_m` and co-rotating rate `omega_t_nom` are
known inputs. A lagged vision sample rewinds through a snapshot ring, updates, and
replays.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ResidualState` | dataclass | State `x`, covariance `P`, and live-measurement flag |
| `ResidualSnapshot` | dataclass | One outer-tick snapshot for rewind |
| `ResidualFilter` | dataclass | Configured `Q`, `R`, and `P0` |
| `predict` | function | Kalman predict with known rate inputs |
| `update` | function | Vision update of elevation error |
| `rewind_update` | function | Delayed update through the snapshot ring |
| `push_snapshot` | function | Append a snapshot and drop the oldest past capacity |

## Inputs and outputs

`ResidualFilter.from_config(cfg, dt_outer_s)` builds scalars from `ResidualConfig`
and the outer period.
`predict` returns a predicted `ResidualState`. `update` takes `z_v` in radians.
`rewind_update` takes the ring, current state, `now`, shutter time, `z_v`, and
horizon.

## Behavior

1. Cold state is `e = 0`, `omega_t_res = 0`, `has_measurement = False`.
2. The first update may snap `e` to `z_v`.
3. Predict runs every outer tick. Update runs when a vision sample arrives.
4. A lagged `z_v` restores the snapshot at shutter time, updates, and replays to
   `now`. Samples older than the rewind horizon are dropped.

## Errors and faults

None.

## Messages

None.

## Configuration

`ResidualConfig.Q_diag`, `R_v`, `P0_diag`, `rewind_horizon_s`, and
`rewind_snapshots`, plus `OuterLoopConfig.dt_s`.

## Constraints

All functions are pure. Outer state is only `(e, omega_t_res)`. `omega_g` is an
input, not a filter state.

## Related documents

- [`flight.payload.tracking`](../tracking.md)
- [`flight.payload.gimbal.outer`](../gimbal/outer.md)
- [`flight.payload.control`](../control.md)
