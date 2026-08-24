# flight.payload.gimbal.lqr

**Source:** `packages/flight/src/flight/payload/gimbal/lqr.py`
**Kind:** pure module

## Purpose

The LQR module computes discrete-time optimal gains and applies the control law to a four-state
error vector.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `LqrController` | class | Holds gain matrix K and max slew clamp |
| `LqrController.from_config` | function | Builds K from `ControllerConfig` via DARE |
| `compute_control` | function | Returns u = -K @ error, clamped |

## Inputs and outputs

`from_config(cfg)` returns an `LqrController` with a `(2, 4)` gain matrix.

`compute_control(controller, state_error)` takes a `(4,)` vector and returns a `(2,)` rate
command in deg/s.

## Behavior

1. `from_config` builds constant-velocity A and B matrices with timestep `kalman_dt_s`.
2. It forms Q and R from `lqr_Q_diag` and `lqr_R_diag`.
3. It solves the discrete algebraic Riccati equation for K.
4. On linear algebra failure, K falls back to proportional gains on pan and tilt position states.
5. `compute_control` computes u = -K @ state_error and clips to `max_slew_deg_s`.

## Errors and faults

None at runtime. DARE failure uses the proportional fallback at construction time.

## Messages

None.

## Configuration

Reads `ControllerConfig`: `kalman_dt_s`, `lqr_Q_diag`, `lqr_R_diag`, `max_slew_deg_s`.

## Constraints

Pure module. The control core negates u when mapping to physical gimbal rates in RATE mode.

## Related documents

- [`flight.payload.gimbal`](gimbal.md)
- [`flight.payload.tracking.kalman`](tracking/kalman.md)
