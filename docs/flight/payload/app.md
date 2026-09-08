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
| `LockGate` | dataclass | Fail-closed launch-lock gate; `UNKNOWN` counts as engaged |
| `SafeLatch` | dataclass | SAFE flag that replaces tracking rate with the stow loop |
| `StowGate` | dataclass | Re-issue STOW after lock release if still SAFE |
| `PoseIntent` | dataclass | Ground STOW / HOME / GOTO waiting for the next outer tick |
| `PayloadApp` | dataclass | Frozen holder of injected services and config slices |
| `PayloadApp.from_config` | static method | Builds the app from `PactConfig` and injected drivers |
| `PayloadApp.poll_mode_changes` | method | Drains `ModeChangeMsg`; returns SAFE entry and exit flags |
| `PayloadApp.poll_lock_state` | method | Drains `LaunchLockStateMsg` and updates the lock gate |
| `PayloadApp.handle_commands` | method | Applies routed STOW / HOME / GOTO into `PoseIntent` |
| `PayloadApp.process_frame` | method | Preprocess, detect, enqueue shutter-stamped vision |
| `PayloadApp.advance_outer` | method | Catch up the outer loop in `T_out` steps |
| `PayloadApp.advance_inner` | method | Catch up the inner loop in `T_in` steps and write torque |
| `PayloadApp.run` | method | Outer loop plus concurrent inner thread until stop |

## Inputs and outputs

`from_config` takes `PactConfig`, `ImagingSensor`, `GimbalActuator`, `IssEphemeris`,
`DetectorBackend`, `MessageBus`, `Clock`, `MosaicCalibration`, and `StorageWriter`.
It returns a `PayloadApp`. It raises `ValueError` on invalid sensor or inference
geometry.

`process_frame` takes a `MosaicFrame` and `ControlState`. It enqueues a vision
sample and does not write torque.

## Behavior

1. `from_config` validates mosaic dimensions, band layout, and inference input
   geometry, then subscribes to mode, launch-lock, and routed-command messages.
   The lock gate starts engaged.
2. `run` starts sensor acquisition and an inner torque thread. All gimbal HAL calls
   are serialized by the actuator I/O lock. Torque commands carry a monotonic
   authority deadline that the driver must enforce independently.
3. Each outer iteration publishes a heartbeat on the watchdog interval, drains mode,
   lock, and pose commands, acquires a frame, and enqueues a shutter-stamped vision
   sample (`theta_g` and ISS at ingest).
4. `advance_outer` stamps a clock origin on the first call (`None` means not
   started). It dequeues a vision sample only when `sample_t <= t`, reads
   ephemeris, and publishes pointing telemetry. Pose `GimbalCommandMsg` is published
   on STOW / HOME / ABSOLUTE after a successful HAL latch. HAL `Err` does not
   publish command-success. Catch-up longer than `catchup_max_s` publishes
   `GIMBAL_FAULT`.
5. Any encoder `Err` immediately requests confirmed inhibit; cached position may
   support outer-loop continuity but cannot authorize torque. Transient
   `COMM_TIMEOUT` failures use a bounded retry budget and reset inner controller
   memory before autonomous recovery. Integrity and unclassified faults latch SAFE.
6. Healthy SAFE operation replaces tracking rate with the stow position loop.
   Launch lock or a latched actuator-integrity fault inhibits torque instead.
   After lock release, a pending SAFE STOW is re-issued.

## Errors and faults

| Fault / error | Trigger |
| --- | --- |
| Preprocessing faults | Calibration, demosaic, or band-select failure |
| Detection faults | Detector returns `Err` |
| Gimbal actuation faults | HAL `Err`, stale feedback, or unconfirmed containment |
| `ValueError` at startup | Invalid sensor mosaic or inference geometry in `from_config` |
| Camera stall | `acquire_frame` returns `Err` |

## Messages

| Direction | Message types |
| --- | --- |
| Subscribe | `ModeChangeMsg`, `LaunchLockStateMsg`, `RoutedCommandMsg` |
| Publish | `HeartbeatMsg`, `InferenceResultMsg`, `GimbalCommandMsg`, `FaultEventMsg`, `TelemetryEventMsg`, `ProductRefMsg`, `CommandAckMsg` |

Vision samples do not travel on the bus.

## Configuration

| Config slice | Use |
| --- | --- |
| `SensorConfig` | Mosaic geometry, bit depth, IFOV, band layout |
| `InferenceConfig` | Input bands and tensor size |
| `PreprocessingConfig` | Quality-flag thresholds and smear budget |
| `FaultConfig` | Heartbeat interval |
| `ControllerConfig` | Nested vision, arbiter, inner, outer, residual, position, and integrity tables |
| `GimbalConfig` | Plant, envelopes, encoder |
| `EphemerisConfig` | WGS-84 and circular-orbit elements |

## Constraints

Preprocessing runs as function calls inside `process_frame`; it never publishes
`ProcessedFrameMsg`. The full selected band plane is passed to inference. Frame
quality follows the inference result for science qualification and does not become
a control safety flag. The app uses `Clock.monotonic_s()` for loops and
`Clock.utc_s()` for ephemeris.

## Related documents

- [`flight.payload.control`](control.md)
- [`flight.payload.preprocess`](preprocess.md)
- [`flight.payload.inference`](inference.md)
- [`flight.payload.calibration_io`](calibration_io.md)
