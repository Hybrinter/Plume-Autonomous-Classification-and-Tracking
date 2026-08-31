# flight.payload.tracking.kalman

**Source:** `packages/flight/src/flight/payload/tracking/kalman.py`
**Kind:** pure module

## Purpose

This module implements a per-axis 4-state linear Kalman filter for gimbal pointing.
The state is boresight error `e`, gimbal angle `theta_g`, target rate `omega_t`, and
gimbal rate `omega_g` in degrees and degrees per second.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `AxisKalmanState` | dataclass | One-axis estimate `x` and covariance `p` |
| `DualKalmanState` | dataclass | Azimuth and elevation `AxisKalmanState` |
| `KalmanFilter` | dataclass | Continuous plant, H, Q, and measurement R |
| `KalmanFilter.from_config` | static method | Builds the plant from `ControllerConfig` |
| `KalmanFilter.initial_axis` | static method | Zero-rate axis state |
| `KalmanFilter.initial_state` | static method | Dual-axis zero-rate state |
| `continuous_plant` | function | Returns `(A_c, B_c)` |
| `discretize` | function | Exact `(Phi, B_d)` for a given `dt` |
| `predict` | function | Propagates one axis under a held rate command |
| `update_vis` | function | Vision update on `e` |
| `update_enc` | function | Encoder update on `theta_g` |

## Inputs and outputs

`predict(kf, state, u, dt)` returns `AxisKalmanState`. `u` is the applied rate in deg/s.

`update_vis` and `update_enc` return `Ok[AxisKalmanState] | Err[FaultCode]`.

## Behavior

1. `from_config` builds `A_c` and `B_c` from `tau_cl_s` with
   `omega_g_dot = (-omega_g + u) / tau_cl_s`.
2. `predict` forms `Phi` and `B_d` from the Van Loan exponential and applies
   `x = Phi x + B_d u`. A `dt` of zero or less returns the input state.
3. `update_vis` uses `H = [1, 0, 0, 0]`. `update_enc` uses `H = [0, 1, 0, 0]`.

## Errors and faults

| Result | Trigger |
| --- | --- |
| `Err(GIMBAL_RUNAWAY)` | Innovation variance S is singular |

## Messages

None.

## Configuration

Uses `ControllerConfig.tau_cl_s`, `kalman_process_noise`, `kalman_r_vis`, and
`kalman_r_enc`.

## Constraints

The two axes are independent copies of the same plant. Process noise on `theta_g` is
kept small. LQR uses the same `A_c` and `B_c`.

## Related documents

- [`flight.payload.tracking`](../tracking.md)
- [`flight.payload.tracking.rewind`](rewind.md)
- [`flight.payload.gimbal.lqr`](../gimbal/lqr.md)
