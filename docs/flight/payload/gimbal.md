# flight.payload.gimbal

**Source:** `packages/flight/src/flight/payload/gimbal`
**Kind:** package

## Purpose

The gimbal package holds pure gimbal control logic: the pointing FSM, LQR law,
boresight geometry, inner rate servo, typed command values, encoder runaway
monitoring, and pre-arbiter safety gates.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`arbiter`](gimbal/arbiter.md) | pure module | IDLE / ACQUIRING / TRACKING / SCAN / SAFE FSM |
| [`lqr`](gimbal/lqr.md) | pure module | Continuous-time LQR rate gain and control output |
| [`pointing`](gimbal/pointing.md) | pure module | Boresight error and displacement from blob centroids |
| [`rate_servo`](gimbal/rate_servo.md) | pure module | Inner rate PI with computed-torque decoupling |
| [`request`](gimbal/request.md) | pure module | Typed gimbal command returned by pure cores |
| [`runaway`](gimbal/runaway.md) | pure module | Encoder rate divergence monitor |
| [`safety`](gimbal/safety.md) | pure module | Confidence, area, deadband, and rate-limit gates |

## Package interface

Re-exports: `ArbiterState`, `GimbalArbiter`, `GimbalRequest`, `INITIAL_RATE_SERVO_STATE`,
`INITIAL_RUNAWAY_STATE`, `RateServoState`, `RunawayState`, `LqrController`,
`apply_confidence_gate`, `apply_min_area_gate`, `boresight_error_deg`, `check_deadband`,
`check_rate_limit`, `check_runaway`, `compute_control`, `rate_servo_step`, `reset_servo`,
`target_displacement_px`.

## Interactions

Pure cores return `GimbalRequest` and `TelemetryEventMsg` values to `PayloadController`
and the app shell. The shell maps requests onto `GimbalActuator` HAL calls and
publishes `GimbalCommandMsg`. No gimbal module accesses the bus or HAL directly.

## Constraints

All modules are pure except that `GimbalArbiter._transition_event` stamps telemetry
with `utc_now_iso()` for event records returned to the caller. `GimbalRequest` never
travels on the bus. STOW and ABSOLUTE commands bypass deadband suppression.

## Related documents

- [`flight.payload.control`](../control.md)
- [`flight.payload.tracking`](../tracking.md)
