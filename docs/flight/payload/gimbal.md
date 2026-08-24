# flight.payload.gimbal

**Source:** `packages/flight/src/flight/payload/gimbal/`
**Kind:** package

## Purpose

The gimbal package holds pure gimbal control logic. It includes the pointing FSM, LQR law,
boresight geometry, typed requests, and safety gates.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`arbiter`](gimbal/arbiter.md) | module | IDLE/ACQUIRING/TRACKING/SCAN/SAFE FSM and command generation |
| [`lqr`](gimbal/lqr.md) | module | Discrete-time LQR gain and control computation |
| [`pointing`](gimbal/pointing.md) | module | Boresight-relative angular error and displacement |
| [`request`](gimbal/request.md) | module | Typed gimbal command value from the pure core |
| [`runaway`](gimbal/runaway.md) | module | Encoder-divergence runaway monitor |
| [`safety`](gimbal/safety.md) | module | Confidence, area, deadband, and rate-limit gates |

## Package interface

Re-exports: `ArbiterState`, `GimbalArbiter`, `GimbalRequest`, `LqrController`,
`compute_control`, `boresight_error_deg`, `target_displacement_px`, `RunawayState`,
`INITIAL_RUNAWAY_STATE`, `check_runaway`, `apply_confidence_gate`, `apply_min_area_gate`,
`check_deadband`, `check_rate_limit`.

## Interactions

None at the bus layer. `GimbalRequest` flows by return value to the app shell. The shell maps
requests to `GimbalActuator` calls and publishes `GimbalCommandMsg`.

## Constraints

All modules are pure. The arbiter stores no mutable instance state; `ArbiterState` is threaded
externally. `GimbalRequest` is not a bus message.

## Related documents

- [`flight.payload`](payload.md)
- [`flight.payload.control`](control.md)
