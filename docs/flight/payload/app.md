# flight.payload.app

**Source:** `packages/flight/src/flight/payload/app.py`
**Kind:** app shell

## Purpose

The payload app shell binds HAL drivers and the pure control core into one acquisition loop. It
runs preprocess, detect, and control as in-process stages on each frame.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `TickOutcome` | class | Per-frame summary: frame id, fault, command flag, gimbal state |
| `LockGate` | class | Mutable launch-lock gate; inhibits gimbal motion when engaged |
| `PayloadApp` | class | Frozen holder of injected services and config slices |
| `PayloadApp.from_config` | function | Builds the app from `PactConfig` and injected drivers |
| `PayloadApp.poll_mode_changes` | method | Drains `ModeChangeMsg`; returns safe entry and clear flags |
| `PayloadApp.poll_lock_state` | method | Drains `LaunchLockStateMsg`; updates the launch-lock gate |
| `PayloadApp.process_frame` | method | Runs one frame: preprocess, detect, control, actuate |
| `PayloadApp.run` | method | Acquisition loop with heartbeats until stop |

## Inputs and outputs

`from_config(cfg, sensor, gimbal, detector, bus, clock, calib, storage)` returns a `PayloadApp`.
It raises `ValueError` on invalid sensor or inference geometry.

`poll_mode_changes()` returns `(safe_commanded, safe_cleared)`.

`poll_lock_state()` returns nothing. It updates `lock_gate.engaged`.

`process_frame(raw, state, now, slew_rate_deg_per_s, gimbal_pos, safe_commanded, safe_cleared)`
returns `(ControlState, TickOutcome)`.

`run(stop_event)` returns nothing. It threads `ControlState` from `initial_state()`.

## Behavior

1. `from_config` validates even mosaic dimensions, band plane size, integer decimation factor,
   `mosaic_layout`, and `input_bands`.
2. `process_frame` calibrates the raw mosaic, separates bands, normalizes, selects bands, and
   computes quality flags on the full plane.
3. In TRACKING with an initialized EMA, it crops a full-resolution ROI around the Kalman target.
   Otherwise it decimates the full plane to the inference input size.
4. It builds a local `ProcessedFrameMsg` and calls `detector.detect()`.
5. On success it publishes `InferenceResultMsg`, stores a mask thumbnail, and calls
   `controller.step()`.
6. It publishes arbiter telemetry and maps an issued `GimbalRequest` to HAL calls. A launch-lock
   gate suppresses motion and emits a telemetry event.
7. `run` emits `HeartbeatMsg` on `watchdog_interval_s`, acquires frames, derives slew rate from
   consecutive encoder reads, and calls `process_frame`. On acquisition failure with SAFE
   commanded, it calls `stow()` directly.

## Errors and faults

| Fault | Trigger |
| --- | --- |
| `FRAME_MALFORMED` | Calibration, demosaic, or band select failure |
| `INFERENCE_NAN` | Non-finite calibrated mosaic |
| Detection faults | Returned by the detector backend |
| Control faults | `GIMBAL_RUNAWAY` from deadband strikes or encoder runaway |
| HAL faults | Gimbal actuation or sensor stall |
| Startup | `ValueError` from `from_config` on bad geometry |

## Messages

Subscribes: `ModeChangeMsg`, `LaunchLockStateMsg`.

Publishes: `HeartbeatMsg`, `InferenceResultMsg`, `TelemetryEventMsg`, `GimbalCommandMsg`,
`FaultEventMsg`, `ProductRefMsg`.

## Configuration

Reads `SensorConfig`, `InferenceConfig`, `PreprocessingConfig`, `FaultConfig`, and
`ControllerConfig` (via `PayloadController.from_config`).

## Constraints

`PayloadApp` is frozen. `LockGate` is mutable and shared. `now` comes from `Clock.monotonic_s()`.
Message timestamps use `Clock.wall_clock_iso()`. The first frame and failed encoder reads use
slew rate 0.0.

## Related documents

- [`flight.payload`](payload.md)
- [`flight.payload.control`](control.md)
- [`flight.payload.preprocess`](preprocess.md)
