# flight.payload.gimbal

**Source:** `packages/flight/src/flight/payload/gimbal`
**Kind:** package

## Purpose

The gimbal package holds pure elevation control logic: the pointing FSM, inner and
outer laws, CoG geometry, pose requests, and pre-arbiter safety gates.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`arbiter`](gimbal/arbiter.md) | pure module | TRACKING / REWIND / SAFE FSM |
| [`inner`](gimbal/inner.md) | pure module | PI plus computed torque |
| [`outer`](gimbal/outer.md) | pure module | Residual feedforward and smear clip |
| [`position`](gimbal/position.md) | pure module | STOW / HOME / GOTO rate into the inner PI |
| [`rate_fit`](gimbal/rate_fit.md) | pure module | Causal polynomial encoder-rate estimator |
| [`intersect`](gimbal/intersect.md) | pure module | Pinhole CoG Earth intersect |
| [`predictor`](gimbal/predictor.md) | pure module | Co-rotating elevation rate |
| [`geo`](gimbal/geo.md) | pure module | Mount, LVLH, and WGS-84 helpers |
| [`pointing`](gimbal/pointing.md) | pure module | Pinhole boresight error |
| [`request`](gimbal/request.md) | pure module | Typed pose command from the pure core |
| [`safety`](gimbal/safety.md) | pure module | Confidence and area gates |

## Package interface

Re-exports: `ArbiterState`, `GimbalArbiter`, `GimbalRequest`, `InnerResult`,
`IntersectResult`, `apply_confidence_gate`, `apply_min_area_gate`,
`boresight_error_deg`, `clip_rate`, `fit_rate`, `inner_step`, `intersect_cog`,
`outer_rate`, `pinhole_error_rad`, `position_rate`, `predict_los`,
`smear_cap_rad_s`, `target_displacement_px`.

## Interactions

Pure cores return `GimbalRequest` and `TelemetryEventMsg` values to
`PayloadController` and the app shell. The shell maps pose requests onto
`GimbalActuator` HAL calls and writes torque from the inner loop. No gimbal module
accesses the bus or HAL directly.

## Constraints

All modules are pure except that `GimbalArbiter._transition_event` stamps telemetry
with `utc_now_iso()` for event records returned to the caller. `GimbalRequest` never
travels on the bus. There is no gimbal azimuth command.

## Related documents

- [`flight.payload.control`](../control.md)
- [`flight.payload.tracking`](../tracking.md)
