# flight.payload

**Source:** `packages/flight/src/flight/payload/`
**Kind:** package

## Purpose

The payload package runs the science pipeline in one app. It acquires mosaic frames, preprocesses
them, runs detection, steps the gimbal controller, and publishes results on the bus.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`app`](payload/app.md) | module | Payload app shell: acquisition loop, frame processing, HAL actuation |
| [`control`](payload/control.md) | module | Pure control core: gates, tracking, arbiter, LQR, safety |
| [`calibration_io`](payload/calibration_io.md) | module | Startup loading of mosaic calibration artifacts |
| [`gimbal`](payload/gimbal.md) | package | Gimbal FSM, control law, pointing math, safety gates |
| [`model`](payload/model.md) | package | Swappable detectors, blob extraction, model verification |
| [`preprocess`](payload/preprocess.md) | package | Pure preprocessing from raw mosaic to inference tensor |
| [`tracking`](payload/tracking.md) | package | EMA smoothing, Kalman filter, blob association |

## Package interface

None. Subpackages expose symbols through their own `__init__.py` files.

## Interactions

The payload app subscribes to `ModeChangeMsg` and `LaunchLockStateMsg`. It publishes
`HeartbeatMsg`, `InferenceResultMsg`, `TelemetryEventMsg`, `GimbalCommandMsg`, `FaultEventMsg`,
and `ProductRefMsg`. It uses the `ImagingSensor`, `GimbalActuator`, and `StorageWriter` HAL
protocols. `ProcessedFrameMsg` stays inside `process_frame()` and is not published.

## Constraints

Preprocessing, detection, and control run as function calls inside one app loop. Apps do not
cross-import other subsystem apps. Pure cores take `now` as an argument and perform no I/O.

## Related documents

- [`flight`](../flight.md)
