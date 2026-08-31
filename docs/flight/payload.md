# flight.payload

**Source:** `packages/flight/src/flight/payload`
**Kind:** package

## Purpose

The payload package runs the science imaging loop. It acquires mosaic frames, preprocesses
them, runs detection, and ingests vision. A control thread or SIL tick runs the inner PI
and outer LQG and drives the gimbal.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`app`](payload/app.md) | app shell | Imaging ingest plus control-thread / SIL torque ticks |
| [`control`](payload/control.md) | pure module | Ingest, outer LQG, and inner rate PI |
| [`calibration_io`](payload/calibration_io.md) | module | Loads checksummed mosaic calibration artifacts at startup |
| [`blobs`](payload/blobs.md) | module | Connected-component blob extraction from a probability mask |
| [`preprocess`](payload/preprocess.md) | package | Pure functions from raw mosaic to inference tensor |
| [`inference`](payload/inference.md) | package | Classifier, segmentor, detector composer, and artifact verification |
| [`gimbal`](payload/gimbal.md) | package | Pure gimbal FSM, control law, pointing math, and safety gates |
| [`tracking`](payload/tracking.md) | package | EMA smoothing, Kalman estimation, and blob association |

## Package interface

The package root has no `__init__.py` re-exports. The composition root imports submodules
directly (`flight.payload.app`, `flight.payload.control`, and the child packages).

## Interactions

The payload app subscribes to `ModeChangeMsg` and `LaunchLockStateMsg`. It publishes
`HeartbeatMsg`, `InferenceResultMsg`, `GimbalCommandMsg`, `FaultEventMsg`,
`TelemetryEventMsg`, and `ProductRefMsg`. It uses the `ImagingSensor`, `GimbalActuator`,
and `StorageWriter` HAL protocols. Preprocessing runs inside `process_frame()` and does
not publish `ProcessedFrameMsg` on the bus.

## Constraints

Preprocessing stays co-located in `PayloadApp.process_frame()` with no bus or thread
boundary before inference. Decision cores (`PayloadController`, `GimbalArbiter`, tracking
and gimbal helpers) are pure: no I/O, no bus access, no clock reads. Apps talk to peer
subsystems only through typed bus messages. Large artifacts (tensors, masks) bypass the
bus; only compact records travel on it.

## Related documents

- [`flight.core.composition`](core/composition.md)
- [`flight.hal.interfaces.sensor`](hal/interfaces/sensor.md)
- [`flight.hal.interfaces.gimbal`](hal/interfaces/gimbal.md)
