# flight.payload

**Source:** `packages/flight/src/flight/payload`
**Kind:** package

## Purpose

The payload package runs the science imaging loop. It acquires mosaic frames, preprocesses
them, runs detection, steps the pure graph and servo cores, and drives the gimbal. The
package holds one app shell and several pure libraries for preprocessing, detection,
tracking, graphs, and gimbal control.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`app`](payload/app.md) | app shell | Activation drain, capture, detect, command drain; inner/outer catch-up |
| [`control`](payload/control.md) | pure module | Mode-free cascaded elevation servo loops |
| [`state`](payload/state.md) | pure module | `PayloadState` threaded runtime record and name helpers |
| [`records`](payload/records.md) | pure module | Compact observation and activation-context value records |
| [`imaging`](payload/imaging.md) | pure module | Planned capture deadlines, duty floor, and inference decimation |
| [`lifecycle`](payload/lifecycle.md) | module | Bounded INIT effect executor and observed lifecycle services |
| [`calibration_io`](payload/calibration_io.md) | module | Loads checksummed mosaic calibration artifacts at startup |
| [`blobs`](payload/blobs.md) | module | Connected-component blob extraction from a probability mask |
| [`preprocess`](payload/preprocess.md) | package | Pure functions from raw mosaic to inference tensor |
| [`inference`](payload/inference.md) | package | Classifier, segmentor, detector composer, and artifact verification |
| [`gimbal`](payload/gimbal.md) | package | Mode-free references, inner/outer laws, pointing math, safety gates |
| [`graphs`](payload/graphs.md) | package | The five pure payload mode graphs and their closed runtime dispatch |
| [`tracking`](payload/tracking.md) | package | Residual Kalman filter and blob association |

## Package interface

The package root has no `__init__.py` re-exports. The composition root imports submodules
directly (`flight.payload.app`, `flight.payload.control`, and the child packages).

## Interactions

The payload app subscribes to `SystemModeActivatedMsg` (the accepted-activation boundary),
`SafetyStateMsg` (fault-owned evidence), `RoutedCommandMsg`, and `FaultEventMsg`
(payload-origin containing faults). It publishes
`HeartbeatMsg`, `InferenceResultMsg`, `GimbalCommandMsg`, `FaultEventMsg`,
`TelemetryEventMsg`, `ProductRefMsg`, `SystemModeRequestMsg`, and
`SystemModeSyncRequestMsg`. It uses the `ImagingSensor`, `GimbalActuator`,
`IssEphemeris`, and `StorageWriter` HAL protocols. Preprocessing runs inside
`process_frame()` and does not publish `ProcessedFrameMsg` on the bus. INIT
effect intents run on one bounded lifecycle worker through
`flight.payload.lifecycle`; the inference runtime holder stays empty until the
control owner installs a verified session.

## Constraints

Preprocessing stays co-located in `PayloadApp.process_frame()` with no bus or thread
boundary before inference. Decision cores (the payload graphs, `ServoController`, tracking
and gimbal helpers) are pure: no I/O, no bus access, no clock reads, and no `SystemMode`
imports inside graphs - only the shell maps accepted activations onto graphs. Apps talk
to peer subsystems only through typed bus messages. Large artifacts (tensors, masks)
bypass the bus; only compact records travel on it.

## Related documents

- [`flight.core.composition`](core/composition.md)
- [`flight.hal.interfaces.sensor`](hal/interfaces/sensor.md)
- [`flight.hal.interfaces.gimbal`](hal/interfaces/gimbal.md)
