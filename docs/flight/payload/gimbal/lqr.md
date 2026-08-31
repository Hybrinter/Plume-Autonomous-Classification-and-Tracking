# flight.payload.gimbal.lqr

**Source:** `packages/flight/src/flight/payload/gimbal/lqr.py`
**Kind:** pure module

## Purpose

This module implements a continuous-time LQR rate law for one gimbal axis. The
design plant is `[e, omega_g]`. `u = -K [e, omega_g]` is the physical inner-loop
rate reference in deg/s.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `LqrController` | dataclass | Holds `K` (1x2) and max slew clamp |
| `LqrController.from_config` | static method | Solves CARE and builds `K` from `ControllerConfig` |
| `lqr_plant` | function | Returns `(A_c, B_c)` for `[e, omega_g]` |
| `compute_axis_control` | function | Returns clamped rate for one axis |
| `compute_control` | function | Returns clamped `[az_rate, el_rate]` |

## Inputs and outputs

`from_config(cfg)` returns an `LqrController`.

`compute_control(controller, x_az, x_el)` takes two `(4,)` state vectors and returns
`(2,)` azimuth and elevation commands in deg/s, clamped to `max_slew_deg_s`.

## Behavior

1. Build `A_c` and `B_c` from `tau_cl_s` via `lqr_plant` on `[e, omega_g]`.
2. Form `Q` from `lqr_Q_diag[0]` and `lqr_Q_diag[3]`, and scalar `R` from the first
   `lqr_R_diag` entry.
3. Solve the continuous algebraic Riccati equation and set `K = R^{-1} B' P`.
4. On solver failure, fall back to a proportional gain on `e`.
5. `compute_control` applies `u = -K [e, omega_g]` per axis and clamps to max slew.

## Errors and faults

None at runtime. CARE failure selects the proportional fallback gain.

## Messages

None.

## Configuration

Uses `ControllerConfig`: `tau_cl_s`, `lqr_Q_diag`, `lqr_R_diag`, `max_slew_deg_s`.

## Constraints

`lqr_Q_diag[0]` weights `e`. `lqr_Q_diag[3]` weights `omega_g`. Target rate is not in
the LQR plant. The command is the physical rate.

## Related documents

- [`flight.payload.tracking.kalman`](../tracking/kalman.md)
- [`flight.payload.control`](../control.md)
