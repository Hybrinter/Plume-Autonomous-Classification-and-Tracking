# flight.payload

**Source:** `packages/flight/src/flight/payload`
**Kind:** package

## Purpose

The payload package runs the science imaging loop. It acquires mosaic frames, preprocesses
them, runs detection, steps the pure control core, and drives the gimbal. The package
holds one app shell and several pure libraries for preprocessing, detection, tracking,
and gimbal control.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`app`](payload/app.md) | app shell | Acquire, preprocess, detect, enqueue vision; inner/outer catch-up |
| [`control`](payload/control.md) | pure module | Cascaded elevation inner and outer loops |

| [`calibration_io`](payload/calibration_io.md) | module | Loads checksummed mosaic calibration artifacts at startup |
| [`blobs`](payload/blobs.md) | module | Connected-component blob extraction from a probability mask |
| [`preprocess`](payload/preprocess.md) | package | Pure functions from raw mosaic to inference tensor |
| [`inference`](payload/inference.md) | package | Classifier, segmentor, detector composer, and artifact verification |
| [`gimbal`](payload/gimbal.md) | package | Pure elevation FSM, inner/outer laws, pointing math, and safety gates |
| [`tracking`](payload/tracking.md) | package | Residual Kalman filter and blob association |

## Package interface

The package root has no `__init__.py` re-exports. The composition root imports submodules
directly (`flight.payload.app`, `flight.payload.control`, and the child packages).

## Interactions

The payload app subscribes to `ModeChangeMsg` and `LaunchLockStateMsg`. It publishes
`HeartbeatMsg`, `InferenceResultMsg`, `GimbalCommandMsg`, `FaultEventMsg`,
`TelemetryEventMsg`, and `ProductRefMsg`. It uses the `ImagingSensor`, `GimbalActuator`,
`IssEphemeris`, and `StorageWriter` HAL protocols. Preprocessing runs inside
`process_frame()` and does not publish `ProcessedFrameMsg` on the bus.

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
