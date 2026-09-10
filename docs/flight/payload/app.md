# flight.payload.app

**Source:** `packages/flight/src/flight/payload/app.py`

**Kind:** app shell

## Purpose

`PayloadApp` binds HAL drivers, the detector, the cascaded
`PayloadController`, and the message bus. One timestamped encoder stream feeds
frame association and the outer residual estimator.

The app supports two actuator paths. The Xeryon production path sends signed
rate commands, including the switch-referenced bounded stow step while SAFE.
The detailed SIL plant path runs the existing inner PI and
torque loop.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `TickOutcome` | dataclass | Per-cycle frame, fault, command, and gimbal summary |
| `LockGate` | dataclass | Fail-closed launch-lock gate |
| `SafeLatch` | dataclass | SAFE flag shared with the inner path |
| `StowGate` | dataclass | Pending SAFE STOW request |
| `PoseIntent` | dataclass | Ground STOW / HOME / GOTO request |
| `EncoderStream` | dataclass | Timestamped samples shared across app paths |
| `PayloadApp` | dataclass | Frozen holder of injected services and config |
| `PayloadApp.from_config` | static method | Builds the app from typed config and drivers |
| `PayloadApp.poll_mode_changes` | method | Drains mode messages |
| `PayloadApp.poll_lock_state` | method | Drains launch-lock messages |
| `PayloadApp.handle_commands` | method | Applies routed pose commands |
| `PayloadApp.process_frame` | method | Preprocesses, detects, and queues vision |
| `PayloadApp.advance_outer` | method | Consumes timestamped outer samples |
| `PayloadApp.advance_inner` | method | Advances the detailed-plant inner path |
| `PayloadApp.run` | method | Runs acquisition and the selected control path |

## Inputs and outputs

`from_config` takes `PactConfig`, HAL drivers, `MessageBus`, `Clock`, calibration,
and storage. It returns a `PayloadApp` and raises `ValueError` for invalid
sensor or inference geometry.

`process_frame` takes a `MosaicFrame` and `ControlState`. It records valid
encoder feedback, creates a frame-ID-bearing vision sample, and does not write
a gimbal command.

## Behavior

1. `from_config` validates mosaic dimensions, band layout, and inference input
   geometry. The lock gate starts engaged.
2. Each valid `GimbalPosition` becomes an `EncoderSample` with device timestamp,
   unwrapped angle, variance, and stable sample ID. The sample is stored in the
   shared encoder stream.
3. `advance_outer` consumes the newest unconsumed sample whose device time
   belongs to the historical tick. A current feedback value is not relabeled
   with an older tick time. A missing sample leaves the tick uncommitted and
   does not fabricate zero displacement.
4. The controller receives the encoder sample, queued shutter-stamped vision,
   navigation state, and any explicit predictor-reference replacement. It
   replays the residual history at the encoder sample time.
5. A feedback read failure invalidates the encoder baseline, clears motion
   authority, and requests confirmed inhibition. Recovery starts a new
   residual checkpoint at the first valid sample.
6. The production rate path checks feedback health, creates a signed rate
   command with an absolute validity deadline, and calls `set_rate`. A failed
   command requests inhibition and latches the actuator fault.
7. The detailed SIL path keeps the inner encoder-rate fit, PI, and torque
   command. Its torque thread is not started when the injected actuator exposes
   the production rate interface.
8. SAFE and launch-lock states inhibit motion. SAFE operation commands the stow
   position loop. A pending SAFE STOW is retried after lock release.
9. Shutdown stops acquisition and joins the detailed-plant thread when one is
   running. The Xeryon adapter shutdown path remains fail-closed.

## Errors and faults

| Fault / error | Trigger |
| --- | --- |
| Preprocessing faults | Calibration, demosaic, or band-select failure |
| Detection faults | Detector returns `Err` |
| Encoder fault | Feedback read error, stale timing, invalid sample, or clock reset |
| Gimbal actuation fault | HAL error, stale feedback, or unconfirmed inhibition |
| `ValueError` at startup | Invalid sensor mosaic or inference geometry |
| Camera stall | `acquire_frame` returns `Err` |
| Catch-up fault | Catch-up exceeds `catchup_max_s` |

Encoder, controller, thermal, watchdog, and timing faults request local stop
and drive inhibition. Serial acknowledgement does not establish physical
inhibition without independent watchdog evidence.

## Messages

| Direction | Message types |
| --- | --- |
| Subscribe | `ModeChangeMsg`, `LaunchLockStateMsg`, `RoutedCommandMsg` |
| Publish | `HeartbeatMsg`, `InferenceResultMsg`, `GimbalCommandMsg`, `FaultEventMsg`, `TelemetryEventMsg`, `ProductRefMsg`, `CommandAckMsg` |

Vision samples stay in the app queue. Residual event records stay in pure
controller state and do not travel on the bus.

## Configuration

| Config slice | Use |
| --- | --- |
| `SensorConfig` | Mosaic geometry, bit depth, IFOV, band layout |
| `InferenceConfig` | Input bands and tensor size |
| `PreprocessingConfig` | Quality flags and smear budget |
| `FaultConfig` | Heartbeat interval |
| `ControllerConfig` | Vision, arbiter, inner, outer, residual, position, and integrity settings |
| `GimbalConfig` | Travel envelope and simulation plant values |
| `XeryonConfig` | Rate quantum, serial transport, timing, duty, feedback, and stow limits |
| `EphemerisConfig` | WGS-84 and circular-orbit elements |

## Constraints

Preprocessing runs inside `process_frame`; it does not publish
`ProcessedFrameMsg`. The app uses `Clock.monotonic_s()` for loop control and
the encoder contract uses mapped device sample time in the same domain.

The Xeryon adapter remains motion-disabled until vendor-source and Python
version audit, independent watchdog tests, bounded stow bench tests, measured
feedback cadence, timing-uncertainty evidence, and thermal-duty verification
complete. Qualification also requires the 32-seed chronological oracle tests,
Monte Carlo innovation coverage, and SIL paired-seed results.

## Related documents

- [`flight.payload.control`](control.md)
- [`flight.payload.tracking`](tracking.md)
- [`flight.payload.preprocess`](preprocess.md)
- [`flight.payload.inference`](inference.md)
- [`flight.payload.calibration_io`](calibration_io.md)
- [`flight.hal.interfaces.gimbal`](../hal/interfaces/gimbal.md)
