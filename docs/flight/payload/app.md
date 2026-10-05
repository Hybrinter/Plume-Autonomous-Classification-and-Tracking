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
| `PayloadApp.capture_once` | method | One planned capture cycle under the committed policy |
| `PayloadApp.process_frame` | method | Preprocesses, detects, and queues vision |
| `PayloadApp.sample_feedback` | method | One control-owned feedback read for the SIL root |
| `PayloadApp.note_gimbal_feedback` | method | Records feedback on a non-capture tick |
| `PayloadApp.run` | method | Runs acquisition and the selected control path |

## Inputs and outputs

`from_config` takes `PactConfig`, HAL drivers, an `InferenceRuntime` holder,
`MessageBus`, `Clock`, calibration, storage, and the composition-root
`activation_epoch`. Optional lifecycle services (self-test, home arrival,
verification) and the effect deadline may be injected; production defaults are
the observed checks plus a verifier that always reports PENDING.
`synchronous_lifecycle` defaults to false. Flight keeps the lazy daemon.
SIL passes true through `build_apps`. `from_config` returns
a `PayloadApp` and raises `ValueError` for invalid sensor or inference
geometry or for a graph/node payload policy that resolves to an invalid
combination; policy validation runs before any camera-policy HAL call.

`control_tick` orders one cycle: `poll_activations`, then `advance_inner`,
then `advance_outer`. `poll_activations` accepts epoch-matching
`SystemModeActivatedMsg` records with a newer sequence; an exact duplicate is
ignored without reentering, a stale sequence never selects a graph, and a
same-key content conflict or wrong epoch faults and latches containment. Every
accepted activation inhibits motion, cold-enters the destination graph's
declared policy, and bumps both revisions. A SAFE activation also latches
containment.

`run` calls `capture_once`, then sleeps for `capture_wait_s`. The sleep is
the time remaining until `next_opportunity_s`, at most one outer period. No
future deadline sleeps one outer period. A deadline already passed sleeps 0.

`capture_once` stamps the capture context and the latest state atomically
under the state lock before any settings, acquisition, or shutter work, then
hands cadence and duty to the pure `plan_capture`: `WAIT` does no I/O,
`DRAIN` releases one buffered frame, `CAPTURE` acquires and runs
`process_frame`. The context is rechecked before and after every potentially
blocking HAL stage; a superseded start stops the stream and captures
nothing. `process_frame` stamps the implicit context under the same lock
before preprocessing and detection. Stale contexts stop new product I/O, and
a final atomic validate-and-commit publishes results only when the
activation, policy revision, and containment generation still match. An
injected `gimbal_pos` associates the frame for shutter/GSD geometry only - it
never mutates the control-owned encoder history.

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
10. The committed policy's `duty_cycle` gates imaging through
    `plan_capture`. Duty 0.5 captures even opportunities; a `DRAIN`
    opportunity calls `drain_frame` and a `WAIT` call touches no HAL. The
    schedule phase resets on activation, policy-revision, or
    containment-generation change. This duty is not
    `gimbal.xeryon.hv_duty_fraction`.
11. `record_capture` decimates inference to the first successful capture and
    every `every_n_frames`-th after. A skipped frame runs no detector and
    publishes no inference, product, or vision record - it is not a
    plume-loss observation. `publish_products=False` keeps inference but
    stores and references nothing; the live frame exposure still feeds
    quality flags and GSD.
12. Imaging policy/HAL failures set `CaptureShell.pending_fault` and publish
    the original `FaultEventMsg`; `_poll_local_faults` consumes it once and
    latches containment through the control owner - capture never calls the
    gimbal. Frame-scoped acquire/drain failures report only while their
    context is still current: a stale completion drops the fault entirely.
    Global camera-control failures (setters, start, stop) always report;
    their physical effects persist across activations.
    Acquisition-stop failures retain the acquisition flag and applied
    settings and remain observable even before activation.
13. A changed exposure or gain applies stop, exposure, gain, start in that
    order; an unchanged policy repeats nothing and a cadence-only revision
    does not cycle the camera. One compact `imaging_policy` telemetry record
    emits per applied policy revision under one context-current check - for a
    fresh start, an already running camera, and a confirmed off policy - and
    never for a forced stop of a still-enabled policy or a superseded
    application.
14. A nonfinite capture time is rejected before any setter, start, or
    acquire. `entry_policy` supplies the activation policy: OPERATE enters on
    TRACKING's configured override; other graphs use their declared off
    policy.
15. INIT effect intents go to the `LifecycleExecutor` under the exact
    activation token after a successful reference commit. Outer ticks poll the
    executor and feed terminal `EffectResult`s plus a current verification to
    the next `runtime.step`; results for a different token, kind, id, or
    cancelled generation drop. Any activation change, containment latch, or
    shutdown cancels the worker first, so a blocked service or SDK call cannot
    stall inhibition or teardown. The previously verified session survives a
    failed replacement load.
16. While OPERATE runs with an empty `InferenceRuntime` (no verified session),
    the reference is forced to inhibit; an inference-enabled policy is forced
    off with one `MODEL_CORRUPT` fault and one SAFE request per activation,
    while a capture-only policy may still acquire frames. `process_frame`
    stamps `model_version` from the installed session identity at capture
    start and drops the result when the runtime identity changes mid-flight;
    installation bumps the policy revision so a same-identity reload still
    invalidates in-flight captures.

## Errors and faults

| Fault / error | Trigger |
| --- | --- |
| Preprocessing faults | Stack, calibration, or band-select failure |
| Detection faults | Detector returns `Err` |
| `GIMBAL_ENCODER_INVALID` | Nonfinite encoder angle or timestamp |
| Encoder fault | Feedback read error, stale timing, or clock reset |
| Gimbal actuation fault | HAL error, stale feedback, or unconfirmed inhibition |
| `ValueError` at startup | Invalid channel layout, inference geometry, or payload policy |
| Imaging policy/HAL fault | Invalid resolved policy or sensor stage `Err` queues control-owned containment |
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
| `PayloadPolicyConfig` | Graph and per-node imaging/inference policy overrides |
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
