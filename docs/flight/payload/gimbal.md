# flight.payload.gimbal

**Source:** `packages/flight/src/flight/payload/gimbal`
**Kind:** package

## Purpose

The gimbal package holds pure elevation control logic: scene
selection, inner and outer laws, CoG geometry, mode-free control references,
safety gates, and the light integrity detector.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`inner`](gimbal/inner.md) | pure module | PI plus computed torque |
| [`outer`](gimbal/outer.md) | pure module | Smear cap, stopping governor, and `RateDecision` primitives |
| [`scene`](gimbal/scene.md) | pure module | CoG / boresight scene prediction entry points |
| [`position`](gimbal/position.md) | pure module | STOW / HOME / GOTO rate into the inner PI |
| [`rate_fit`](gimbal/rate_fit.md) | pure module | Causal polynomial encoder-rate estimator |
| [`intersect`](gimbal/intersect.md) | pure module | Pinhole CoG and boresight height-ellipsoid intersect |
| [`footprint`](gimbal/footprint.md) | pure module | Pixel and tile GSD from the camera and orbit geometry |
| [`predictor`](gimbal/predictor.md) | pure module | Co-rotating elevation and unactuated azimuth rates |
| [`geo`](gimbal/geo.md) | pure module | Mount, LVLH, and WGS-84 helpers |
| [`pointing`](gimbal/pointing.md) | pure module | Pinhole boresight error |
| [`request`](gimbal/request.md) | pure module | Typed pose command and mode-free control references |
| [`safety`](gimbal/safety.md) | pure module | Confidence and area gates |
| [`integrity`](gimbal/integrity.md) | pure module | NaN and encoder-freeze detector |

## Package interface

Re-exports: `CameraGeometry`, `ControlReference`,
`InhibitReference`, `InnerResult`,
`IntegrityResult`, `LosPrediction`, `PoseReference`, `RateDecision`,
`RateReference`, `RayHit`, `SceneEstimate`, `SceneSource`, `StowReference`,
`TravelEnvelope`, `apply_confidence_gate`, `apply_min_area_gate`,
`boresight_error_deg`, `boresight_scene`, `check_integrity`, `clip_rate`,
`cog_scene`, `fit_rate`, `fit_rate_timed`, `inner_step`, `intersect_boresight`,
`intersect_cog`, `pinhole_error_rad`, `position_rate`, `predict_los`,
`rate_decision`, `smear_cap_rad_s`, `target_displacement_px`,
`validate_reference`.

## Interactions

Pure cores return `RateDecision`, `ControlReference`, and telemetry records to
the payload graphs and the app shell. The shell maps references onto
`GimbalActuator` HAL calls and writes torque from the inner loop. No gimbal
module accesses the bus or HAL directly.

## Constraints

All modules are pure and mode-free: reference records carry numeric envelopes,
not mode or graph names. There is no gimbal azimuth command.

## Related documents

- [`flight.payload.control`](../control.md)
- [`flight.payload.tracking`](../tracking.md)
