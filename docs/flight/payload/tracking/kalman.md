# flight.payload.tracking.kalman

**Source:** `packages/flight/src/flight/payload/tracking/kalman.py`
**Kind:** pure module

## Purpose

This module implements a 2-D constant-velocity Kalman filter for gimbal pointing
state. The state vector is pan position, tilt position, pan rate, and tilt rate in
degrees and degrees per second.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `KalmanState` | dataclass | State estimate x and covariance P |
| `KalmanFilter` | dataclass | F, H, Q, R matrices |
| `KalmanFilter.from_config` | static method | Builds matrices from `ControllerConfig` |
| `KalmanFilter.initial_state` | static method | Zero-velocity initial state at given position |
| `predict` | function | Propagates state one timestep |
| `update` | function | Incorporates a (pan, tilt) observation |

## Inputs and outputs

`predict(kf, state)` returns `KalmanState`.

`update(kf, state, observation)` takes a `(2,)` observation and returns
`Ok[KalmanState] | Err[FaultCode]`.

## Behavior

1. `from_config` builds a constant-velocity F matrix from `kalman_dt_s`, observation
   matrix H selecting position states, and diagonal Q and R from process and measurement
   noise scalars.
2. `predict` computes `x_pred = F @ x` and `P_pred = F @ P @ F' + Q`.
3. `update` forms innovation covariance S, inverts S for Kalman gain K, and updates x
   and P.

## Errors and faults

| Result | Trigger |
| --- | --- |
| `Err(GIMBAL_RUNAWAY)` | Innovation covariance S is singular |

## Messages

None.

## Configuration

Uses `ControllerConfig.kalman_dt_s`, `kalman_process_noise`, and
`kalman_measurement_noise`.

## Constraints

The filter runs every frame with predict; update runs only when the EMA is initialized.
LQR shares the same timestep and plant structure via `LqrController.from_config`.

## Related documents

- [`flight.payload.tracking`](../tracking.md)
- [`flight.payload.gimbal.lqr`](../gimbal/lqr.md)
