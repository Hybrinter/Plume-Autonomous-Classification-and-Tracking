# flight.payload.tracking.kalman

**Source:** `packages/flight/src/flight/payload/tracking/kalman.py`
**Kind:** pure module

## Purpose

The Kalman module implements a constant-velocity filter for pan and tilt in degree space. State
includes position and rate on each axis.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `KalmanState` | class | State vector x and covariance P |
| `KalmanFilter` | class | F, H, Q, R matrices |
| `KalmanFilter.from_config` | function | Builds matrices from config |
| `KalmanFilter.initial_state` | function | Zero-rate initial state at given position |
| `predict` | function | Propagates state one timestep |
| `update` | function | Incorporates a 2-D observation |

## Inputs and outputs

State vector: `[pan_deg, tilt_deg, pan_rate_deg_s, tilt_rate_deg_s]`.

Observation: `[pan_deg, tilt_deg]`.

`predict(kf, state)` returns `KalmanState`.

`update(kf, state, observation)` returns `Ok(KalmanState)` or `Err(GIMBAL_RUNAWAY)`.

## Behavior

1. `from_config` builds F with timestep `kalman_dt_s`, observation H on positions, and scalar Q
   and R from config noise values.
2. `predict` applies `x = F @ x` and `P = F @ P @ F.T + Q`.
3. `update` computes innovation covariance S, Kalman gain K, and corrected x and P.
4. Singular S returns `Err(GIMBAL_RUNAWAY)`.

The control core predicts every frame and updates only when EMA is initialized.

## Errors and faults

| Result | Trigger |
| --- | --- |
| `Err(GIMBAL_RUNAWAY)` | Singular innovation covariance |

## Messages

None.

## Configuration

Reads `ControllerConfig.kalman_dt_s`, `kalman_process_noise`, `kalman_measurement_noise`.

## Constraints

Pure module. Shares the constant-velocity plant model with `LqrController`.

## Related documents

- [`flight.payload.tracking`](tracking.md)
- [`flight.payload.gimbal.lqr`](gimbal/lqr.md)
- [`flight.payload.control`](control.md)
