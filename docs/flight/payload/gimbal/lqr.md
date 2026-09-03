# flight.payload.gimbal.lqr

**Source:** `packages/flight/src/flight/payload/gimbal/lqr.py`
**Kind:** pure module

## Purpose

This module implements a discrete-time LQR controller for gimbal axis tracking. It
precomputes gain matrix K from the same constant-velocity plant model as the Kalman
filter and computes rate commands from the state error vector.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `LqrController` | dataclass | Holds K matrix and max slew clamp |
| `LqrController.from_config` | static method | Solves DARE and builds K from `ControllerConfig` |
| `compute_control` | function | Returns clamped control vector u = -K @ error |

## Inputs and outputs

`from_config(cfg, max_slew_deg_s)` returns an `LqrController`.

`compute_control(controller, state_error)` takes a `(4,)` error vector and returns `(2,)`
pan and tilt commands in deg/s, clamped to `max_slew_deg_s`.

## Behavior

1. Build state transition A and input B matrices from `kalman_dt_s`.
2. Form Q and R from `lqr_Q_diag` and `lqr_R_diag`.
3. Solve the discrete algebraic Riccati equation for P, then compute
   `K = inv(R + B'PB) B'PA`.
4. On solver failure, fall back to diagonal proportional gains on pan and tilt position
   states.
5. `compute_control` applies `u = -K @ state_error` and clamps to max slew.

## Errors and faults

None at runtime. DARE failure selects the proportional fallback gains.

## Messages

None.

## Configuration

Uses `ControllerConfig`: `kalman_dt_s`, `lqr_Q_diag`, `lqr_R_diag`. Hardware slew clamp
is `max_slew_deg_s` from `GimbalConfig.max_hw_slew_rate_deg_per_s`.

## Constraints

`PayloadController` negates LQR output when mapping to physical slew rates in RATE
mode. LQR runs only when the arbiter issues RATE and the EMA is initialized.

## Related documents

- [`flight.payload.tracking.kalman`](../tracking/kalman.md)
- [`flight.payload.control`](../control.md)
