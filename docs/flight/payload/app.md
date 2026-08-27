# flight.payload.app

**Source:** `packages/flight/src/flight/payload/app.py`
**Kind:** app shell

## Purpose

`PayloadApp` is the payload subsystem app shell. It binds HAL drivers, the detector, the
pure `PayloadController`, and the message bus into one acquisition loop. Each frame it
preprocesses a raw mosaic, runs detection, steps control, and actuates the gimbal when
a command is issued.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `TickOutcome` | dataclass | Per-frame summary: frame id, fault, command flag, gimbal state |
| `LockGate` | dataclass | Mutable launch-lock gate that inhibits gimbal motion when engaged |
| `PayloadApp` | dataclass | Frozen holder of injected services and config slices |
| `PayloadApp.from_config` | static method | Builds the app from `PactConfig` and injected drivers |
| `PayloadApp.poll_mode_changes` | method | Drains `ModeChangeMsg`; returns SAFE entry and exit flags |
| `PayloadApp.poll_lock_state` | method | Drains `LaunchLockStateMsg` and updates the lock gate |
| `PayloadApp.process_frame` | method | Runs one frame end-to-end |
| `PayloadApp.run` | method | Acquisition loop with heartbeats until stop |

## Inputs and outputs

`from_config` takes `PactConfig`, `ImagingSensor`, `GimbalActuator`, `DetectorBackend`,
`MessageBus`, `Clock`, `MosaicCalibration`, and `StorageWriter`. It returns a
`PayloadApp`. It raises `ValueError` on invalid sensor or inference geometry.

`process_frame` takes a `MosaicFrame`, `ControlState`, monotonic `now`, optional slew
rate and gimbal position, and SAFE flags. It returns `(ControlState, TickOutcome)`.

`run` takes a `threading.Event` stop signal and runs until it is set.

## Behavior

1. `from_config` validates mosaic dimensions, band layout, and inference input geometry,
   then subscribes to mode and launch-lock messages.
2. `run` starts sensor acquisition, initializes control state, and loops until stop.
3. Each loop iteration publishes a heartbeat on the watchdog interval, drains mode and
   lock messages, acquires a frame, reads gimbal position, and computes slew rate from
   consecutive encoder reads.
4. `process_frame` runs preprocessing in order: calibrate, demosaic, normalize, select
   bands, quality flags, then mode-dependent ROI (decimated full plane in search,
   Kalman-centered crop in TRACKING).
5. The detector runs on the processed tensor. On success the app publishes
   `InferenceResultMsg`, stores a mask thumbnail, and calls `PayloadController.step`.
6. Telemetry events publish to the bus. Control faults publish `FaultEventMsg`.
7. When a gimbal request exists and the launch lock is not engaged, the app maps the
   request to HAL calls (`set_rate`, `goto_angle`, `stow`, or `home`) and publishes
   `GimbalCommandMsg`. When the lock is engaged, motion is suppressed and a telemetry
   event records the inhibit.
8. On acquisition failure the app publishes a sensor stall fault. If SAFE was commanded
   during the failure, it calls `stow()` directly.

## Errors and faults

| Fault / error | Trigger |
| --- | --- |
| Preprocessing faults | Calibration, demosaic, or band-select failure |
| Detection faults | Detector returns `Err` |
| Control faults | Deadband strike limit or encoder runaway from the controller |
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
| `InferenceConfig` | Input bands and tensor size |
| `PreprocessingConfig` | Quality-flag thresholds |
| `FaultConfig` | Heartbeat interval |
| `ControllerConfig` | Passed to `PayloadController.from_config` |

## Constraints

Preprocessing runs as function calls inside `process_frame`; it never publishes
`ProcessedFrameMsg`. The app uses `Clock.monotonic_s()` for arbiter timing and
`Clock.wall_clock_iso()` for message timestamps. Mask products store through
`StorageWriter` and advertise via compact `ProductRefMsg` records.

## Related documents

- [`flight.payload.control`](control.md)
- [`flight.payload.preprocess`](preprocess.md)
- [`flight.payload.inference`](inference.md)
- [`flight.payload.calibration_io`](calibration_io.md)
