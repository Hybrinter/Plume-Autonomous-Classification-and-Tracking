# flight.payload.app

**Source:** `packages/flight/src/flight/payload/app.py`
**Kind:** app shell

## Purpose

`PayloadApp` is the payload subsystem app shell. It binds HAL drivers, the detector, the
pure `PayloadController`, and the message bus. Imaging preprocesses a raw mosaic, runs
detection, and ingests vision. A control thread (or SIL `apply_control`) runs inner
torque ticks and outer LQG ticks.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `TickOutcome` | dataclass | Per-frame summary: frame id, fault, command flag, gimbal state |
| `LockGate` | dataclass | Mutable launch-lock gate that inhibits gimbal motion when engaged |
| `PayloadApp` | dataclass | Frozen holder of injected services and config slices |
| `PayloadApp.from_config` | static method | Builds the app from `PactConfig` and injected drivers |
| `PayloadApp.poll_mode_changes` | method | Drains `ModeChangeMsg`; returns SAFE entry and exit flags |
| `PayloadApp.poll_lock_state` | method | Drains `LaunchLockStateMsg` and updates the lock gate |
| `PayloadApp.process_frame` | method | Preprocess, detect, ingest vision |
| `PayloadApp.apply_control` | method | Outer LQG plus inner PI; stow or `set_torque` |
| `PayloadApp.run` | method | Imaging loop plus control thread until stop |

## Inputs and outputs

`from_config` takes `PactConfig`, `ImagingSensor`, `GimbalActuator`, `DetectorBackend`,
`MessageBus`, `Clock`, `MosaicCalibration`, and `StorageWriter`. It returns a
`PayloadApp`. It raises `ValueError` on invalid sensor or inference geometry.

`process_frame` takes a `MosaicFrame`, `ControlState`, monotonic `now`, optional slew
rate and gimbal position, and SAFE flags. It returns `(ControlState, TickOutcome)`.
`command_issued` is always false on this path.

`apply_control` takes `ControlState`, `now`, `dt`, and SAFE flags. It returns
`(ControlState, TickOutcome)`.

`run` takes a `threading.Event` stop signal and runs until it is set.

## Behavior

1. `from_config` requires even mosaic dimensions and band-plane size equal to the
   inference input, then subscribes to mode and launch-lock messages.
2. `run` starts acquisition and a daemon control thread, then loops until stop.
3. Each imaging iteration publishes a heartbeat on the watchdog interval, drains mode
   and lock messages, acquires a frame, and computes slew rate from encoder reads.
4. `process_frame` runs calibrate, demosaic, normalize, select bands, and quality flags
   on the full plane. The inference tensor is that plane with crop `(0, 0)` and scale 1.
5. The detector runs on the tensor. On success the app publishes `InferenceResultMsg`,
   stores a mask thumbnail, and calls `ingest_vision` with `capture_monotonic_s`.
6. The control thread waits `dt_inner_min_s`. It reads the encoder, steps the inner PI,
   and calls `set_torque`. On the outer period it also runs `step_outer`.
7. `apply_control` is the single-thread stand-in: one outer tick, then inner chunks, then
   `set_torque` or `stow`. Launch lock suppresses torque and publishes an inhibit event.
8. On acquisition failure the app publishes a sensor stall fault. If SAFE was commanded
   during the failure, it calls `stow()` and resets the PI integrator.

## Errors and faults

| Fault / error | Trigger |
| --- | --- |
| Preprocessing faults | Calibration, demosaic, or band-select failure |
| Detection faults | Detector returns `Err` |
| `FRAME_MALFORMED` | Band-plane size does not equal the inference input |
| Gimbal actuation faults | HAL call returns `Err` |
| `ValueError` at startup | Invalid sensor mosaic or inference geometry in `from_config` |
| Camera stall | `acquire_frame` returns `Err` |

## Messages

| Direction | Message types |
| --- | --- |
| Subscribe | `ModeChangeMsg`, `LaunchLockStateMsg` |
| Publish | `HeartbeatMsg`, `InferenceResultMsg`, `GimbalCommandMsg`, `FaultEventMsg`, `TelemetryEventMsg`, `ProductRefMsg` |

## Configuration

| Config slice | Use |
| --- | --- |
| `SensorConfig` | Mosaic geometry, bit depth, IFOV, band layout |
| `InferenceConfig` | Input bands and tensor size (512, equal to the band plane) |
| `PreprocessingConfig` | Quality-flag thresholds |
| `FaultConfig` | Heartbeat interval |
| `ControllerConfig` | Passed to `PayloadController.from_config`; inner/outer Δt |
| `GimbalConfig` | Inertia and travel, passed to `PayloadController.from_config` |

## Constraints

Preprocessing runs as function calls inside `process_frame`; it never publishes
`ProcessedFrameMsg`. Imaging does not call `set_torque`. The control thread and SIL
`apply_control` share the same pure ticks. Mask products store through `StorageWriter`
and advertise via compact `ProductRefMsg` records.

## Related documents

- [`flight.payload.control`](control.md)
- [`flight.payload.preprocess`](preprocess.md)
- [`flight.payload.inference`](inference.md)
- [`flight.payload.calibration_io`](calibration_io.md)
