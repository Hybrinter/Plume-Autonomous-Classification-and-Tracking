# flight.payload.app

**Source:** `packages/flight/src/flight/payload/app.py`
**Kind:** app shell

## Purpose

`PayloadApp` is the payload subsystem app shell. It binds HAL drivers, the detector,
the cascaded `PayloadController`, and the message bus. Vision samples go to an
in-process queue. The outer loop writes `r`. The inner loop writes torque.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `TickOutcome` | dataclass | Per-cycle summary: frame id, fault, command flag, gimbal state |
| `LockGate` | dataclass | Mutable launch-lock gate that inhibits gimbal motion when engaged |
| `PayloadApp` | dataclass | Frozen holder of injected services and config slices |
| `PayloadApp.from_config` | static method | Builds the app from `PactConfig` and injected drivers |
| `PayloadApp.poll_mode_changes` | method | Drains `ModeChangeMsg`; returns SAFE entry and exit flags |
| `PayloadApp.poll_lock_state` | method | Drains `LaunchLockStateMsg` and updates the lock gate |
| `PayloadApp.process_frame` | method | Preprocess, detect, enqueue vision |
| `PayloadApp.advance_outer` | method | Catch up the outer loop in `T_out` steps |
| `PayloadApp.advance_inner` | method | Catch up the inner loop in `T_in` steps and write torque |
| `PayloadApp.run` | method | Outer loop plus inner thread until stop |

## Inputs and outputs

`from_config` takes `PactConfig`, `ImagingSensor`, `GimbalActuator`, `IssEphemeris`,
`DetectorBackend`, `MessageBus`, `Clock`, `MosaicCalibration`, and `StorageWriter`.
It returns a `PayloadApp`. It raises `ValueError` on invalid sensor or inference
geometry.

`process_frame` takes a `MosaicFrame` and `ControlState`. It enqueues a vision
sample and does not write torque.

## Behavior

1. `from_config` validates mosaic dimensions, band layout, and inference input
   geometry, then subscribes to mode and launch-lock messages.
2. `run` starts sensor acquisition, starts an inner torque thread, and loops until
   stop.
3. Each outer iteration publishes a heartbeat on the watchdog interval, drains mode
   and lock messages, acquires a frame, and enqueues vision.
4. `advance_outer` dequeues a vision sample when its shutter time is due, reads
   ephemeris, and publishes pointing telemetry. Pose `GimbalCommandMsg` is published
   on STOW / HOME / ABSOLUTE. When the launch lock is engaged, published `r` is 0.
5. `advance_inner` reads the encoder and calls `set_torque` unless the launch lock
   is engaged.
6. Catch-up methods step in `T_in` / `T_out` so a `ManualClock` jump still moves the
   plant.

## Errors and faults

| Fault / error | Trigger |
| --- | --- |
| Preprocessing faults | Calibration, demosaic, or band-select failure |
| Detection faults | Detector returns `Err` |
| Gimbal actuation faults | HAL call returns `Err` |
| `ValueError` at startup | Invalid sensor mosaic or inference geometry in `from_config` |
| Camera stall | `acquire_frame` returns `Err` |

## Messages

| Direction | Message types |
| --- | --- |
| Subscribe | `ModeChangeMsg`, `LaunchLockStateMsg` |
| Publish | `HeartbeatMsg`, `InferenceResultMsg`, `GimbalCommandMsg`, `FaultEventMsg`, `TelemetryEventMsg`, `ProductRefMsg` |

Vision samples do not travel on the bus.

## Configuration

| Config slice | Use |
| --- | --- |
| `SensorConfig` | Mosaic geometry, bit depth, IFOV, band layout |
| `InferenceConfig` | Input bands and tensor size |
| `PreprocessingConfig` | Quality-flag thresholds and smear budget |
| `FaultConfig` | Heartbeat interval |
| `ControllerConfig` | Nested vision, arbiter, inner, outer, residual, and position tables |
| `GimbalConfig` | Plant, envelopes, encoder |
| `EphemerisConfig` | WGS-84 and circular-orbit elements |

## Constraints

Preprocessing runs as function calls inside `process_frame`; it never publishes
`ProcessedFrameMsg`. The full selected band plane is passed to inference. The app
uses `Clock.monotonic_s()` for loops and `Clock.utc_s()` for ephemeris.

## Related documents

- [`flight.payload.control`](control.md)
- [`flight.payload.preprocess`](preprocess.md)
- [`flight.payload.inference`](inference.md)
- [`flight.payload.calibration_io`](calibration_io.md)
