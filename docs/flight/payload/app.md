# flight.payload.app

**Source:** `packages/flight/src/flight/payload/app.py`

**Kind:** app shell

## Purpose

`PayloadApp` is the single control owner for the payload: it drains accepted
activations and fault-owned safety evidence, runs the command queue, ticks
the active payload graph, executes the mode-free servo, and drives the HAL.
One timestamped encoder stream feeds frame association and the graph tick
inputs. Capture and inference may block but never mutate graph or servo
state.

The app supports two actuator paths. The Xeryon production path sends signed
rate commands, including the switch-referenced bounded stow step. The
detailed SIL plant path runs the inner PI and torque loop.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `TickOutcome` | dataclass | Per-cycle frame, fault, command, and graph summary |
| `ReferenceCommit` | dataclass | Reference commit result: state, audit flag, HAL fault |
| `ContainmentState` | dataclass | Local inhibit latch plus recovery bookkeeping |
| `ActuatorSafety` | dataclass | Driver evidence and transient retry accounting |
| `EncoderStream` | dataclass | Timestamped samples shared across app paths |
| `PayloadApp` | dataclass | Frozen holder of injected services and config |
| `PayloadApp.from_config` | static method | Builds the app from typed config, drivers, and epoch |
| `PayloadApp.initial_state` | method | Unactivated state with the boot inhibit reference |
| `PayloadApp.poll_activations` | method | Drains activation and safety evidence |
| `PayloadApp.control_tick` | method | One control-owner cycle shared by run and SIL |
| `PayloadApp.handle_commands` | method | Deterministic command-drain seam for tests |
| `PayloadApp.advance_inner` | method | Advances the inner servo path |
| `PayloadApp.advance_outer` | method | Ticks the graph, drains commands, commits outcomes |
| `PayloadApp.capture_once` | method | One policy-gated acquisition cycle |
| `PayloadApp.process_frame` | method | Preprocesses, detects, and queues vision |
| `PayloadApp.sample_feedback` | method | One control-owned feedback read for the SIL root |
| `PayloadApp.note_gimbal_feedback` | method | Records feedback on a non-capture tick |
| `PayloadApp.run` | method | Runs acquisition and the selected control path |

## Inputs and outputs

`from_config` takes `PactConfig`, HAL drivers, `MessageBus`, `Clock`,
calibration, storage, and the composition-root `activation_epoch`. It returns
a `PayloadApp` and raises `ValueError` for invalid sensor or inference
geometry.

`control_tick` orders one cycle: `poll_activations`, then `advance_inner`,
then `advance_outer`. `poll_activations` accepts epoch-matching
`SystemModeActivatedMsg` records with a newer sequence; an exact duplicate is
ignored without reentering, a stale sequence never selects a graph, and a
same-key content conflict or wrong epoch faults and latches containment. Every
accepted activation inhibits motion, cold-enters the destination graph's
declared policy, and bumps both revisions. A SAFE activation also latches
containment.

`capture_once` stamps the capture context and the latest state atomically
under the state lock before any settings, acquisition, or shutter work.
`process_frame` stamps the implicit context under the same lock before
preprocessing and detection. Stale contexts stop new product I/O, and a final
atomic validate-and-commit publishes results only when the activation, policy
revision, and containment generation still match. An injected `gimbal_pos`
associates the frame for shutter/GSD geometry only - it never mutates the
control-owned encoder history.

`sample_feedback` is the thin public seam the SIL composition root calls to
take one final actual-clock feedback sample before a shutter binding.

## Behavior

1. Activation identity is `(epoch, sequence)`. Same-mode newer-sequence
   activations re-inhibit and cold-enter the destination graph. A same-key
   content conflict or a wrong epoch faults and latches containment; a
   sequence gap is accepted with `activation_gap` telemetry; exact duplicates
   and stale sequences are ignored without reentering or latching.
2. One directed graph edge per outer tick: a committed routed command edge
   suppresses the same-tick graph step. Distinct queued commands survive to
   later ticks; duplicates never reexecute.
3. `advance_outer` consumes the newest unconsumed encoder sample whose device
   time belongs to the tick. A missing sample leaves the tick uncommitted and
   does not fabricate zero displacement.
4. Flagged vision is filtered through the pure `accept_vision` contract:
   stale, duplicate, or future flagged samples neither fault nor block a
   valid command; a fresh accepted flagged sample inhibits before command
   handling.
5. `_commit_outcome` processes graph faults and system requests before any
   reference action, then defers graph telemetry events until the reference
   commit succeeds. A failed reference commit emits no committed-edge events.
6. `_commit_reference` arms stow/goto HAL metadata for `StowReference` /
   `PoseReference`. Any metadata failure records the actuator failure,
   latches containment, publishes the HAL fault, and returns
   `ReferenceCommit(fault=code)` with no audit command. The routed command
   then ACKs `REJECTED` with `GIMBAL_FAULT`, restores the original graph and
   hold memory, installs `InhibitReference("actuator failure")`, and disables
   the policy.
7. Catch-up cap violations on either loop latch containment immediately
   alongside the fault; invalid or nonfinite references inhibit, latch, and
   fault before any actuator write.
8. A nonfinite encoder position or timestamp is rejected before it enters the
   encoder history: `Err(GIMBAL_ENCODER_INVALID)`, latch, and fault. A read
   error latches containment through the actuator-failure path.
9. Recovery requires an authorized activation: matching request ID, matching
   epoch, strictly increasing evidence sequence, finite nonfuture observation
   time, no active faults or SAFE latch, fresh confirmed hardware feedback,
   and a fresh valid bounded encoder sample. Evidence is consumed once; a
   replayed request ID cannot release a new latch, and unsafe evidence in the
   same drain always wins over a later clear record.
10. `sensor.capture.duty_cycle` gates imaging. Duty 0.5 captures even
    opportunities. A skipped opportunity calls `drain_frame` and still steps
    the outer loop. This duty is not `gimbal.xeryon.hv_duty_fraction`.
11. Acquisition-stop failures are observable: `_stop_acquisition` retains the
    acquisition flag and applied settings on failure and publishes a fault;
    `HAL.shutdown` errors publish too.

## Errors and faults

| Fault / error | Trigger |
| --- | --- |
| Preprocessing faults | Stack, calibration, or band-select failure |
| Detection faults | Detector returns `Err` |
| `GIMBAL_ENCODER_INVALID` | Nonfinite encoder angle or timestamp |
| Encoder fault | Feedback read error, stale timing, or clock reset |
| Gimbal actuation fault | HAL error, stale feedback, or unconfirmed inhibition |
| `ValueError` at startup | Invalid channel layout or inference geometry |
| Camera stall | `acquire_frame` returns `Err` |
| Camera buffer drain | `drain_frame` returns `Err` |
| Catch-up fault | Catch-up exceeds `catchup_max_s` |
| Actuator reference fault | stow/goto HAL metadata `Err` |

Encoder, controller, thermal, watchdog, and timing faults request local stop
and drive inhibition. Serial acknowledgement does not establish physical
inhibition without independent watchdog evidence.

## Messages

| Direction | Message types |
| --- | --- |
| Subscribe | `SystemModeActivatedMsg`, `SafetyStateMsg`, `RoutedCommandMsg`, `FaultEventMsg` (payload-local containment) |
| Publish | `HeartbeatMsg`, `InferenceResultMsg`, `GimbalCommandMsg`, `FaultEventMsg`, `TelemetryEventMsg`, `ProductRefMsg`, `CommandAckMsg`, `SystemModeRequestMsg`, `SystemModeSyncRequestMsg` |

Vision samples stay in the app queue. Residual event records stay in pure
graph state and do not travel on the bus.

## Configuration

| Config slice | Use |
| --- | --- |
| `SensorConfig` | Mosaic geometry, bit depth, IFOV, band layout, capture duty |
| `InferenceConfig` | Input bands, sensor size, tile grid, and reference GSD |
| `PreprocessingConfig` | Quality flags and smear budget |
| `FaultConfig` | Heartbeat and watchdog intervals |
| `ControllerConfig` | Vision, operate, inner, outer, residual, position, integrity, predictor |
| `GimbalConfig` | Travel envelope and simulation plant values |
| `XeryonConfig` | Rate quantum, serial transport, timing, duty, feedback, and stow limits |
| `EphemerisConfig` | WGS-84 and circular-orbit elements |

## Constraints

The launch restraint is a strap. The crew removes it before commissioning. The
crew installs it for return. Flight software does not sense or command the
strap. SAFE never issues pose commands; it inhibits motion only.

Preprocessing runs inside `process_frame`; it does not publish
`ProcessedFrameMsg`. The app uses `Clock.monotonic_s()` for loop control and
the encoder contract uses mapped device sample time in the same domain. The
SIL composition root owns clock advancement; `control_tick` only reads it.

The Xeryon adapter remains motion-disabled until vendor-source and Python
version audit, independent watchdog tests, bounded stow bench tests, measured
feedback cadence, timing-uncertainty evidence, and thermal-duty verification
complete. Qualification also requires the 32-seed chronological oracle tests,
Monte Carlo innovation coverage, and SIL paired-seed results.

## Related documents

- [`flight.payload.state`](state.md)
- [`flight.payload.control`](control.md)
- [`flight.payload.graphs`](graphs.md)
- [`flight.payload.tracking`](tracking.md)
- [`flight.payload.preprocess`](preprocess.md)
- [`flight.payload.inference`](inference.md)
- [`flight.payload.calibration_io`](calibration_io.md)
- [`flight.hal.interfaces.gimbal`](../hal/interfaces/gimbal.md)
