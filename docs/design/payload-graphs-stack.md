# Payload graph architecture stack

**Status:** architecture approved; pure foundations and runtime cutover implemented.
This brief records settled design for a multi-PR implementation stack. The live payload
runtime uses authoritative activations; configurable imaging and INIT effects remain
later stack work. The teammate-owned
system-mode authority interface is recorded here as agreed scope, but teammate confirmation of
the shared contract is a pending external dependency; nothing below is implemented authority
code and nothing here asserts their agreement.

## Purpose

Build a statically typed, system-activation-driven payload graph architecture in eight
sequential, reviewable PRs, preserving pure decision cores, independent actuator containment,
and graph-owned imaging and inference policy.

## Confirmed decisions

1. The system modes are exactly `IDLE`, `STOW`, `SAFE`, `INIT`, and `OPERATE`; enum names and
   string values mirror one another. Remove `ACTIVE`, `SCAN`, `MODEL_UPLINK`, and
   `DATA_DOWNLINK`.
2. Payload graph names mirror those five modes. System authority chooses a mode; payload graphs
   define payload behavior. Graph/node functions never branch on or import `SystemMode`.
3. This stack owns shared enums and bus contracts. The teammate owns system-mode authority.
4. Requests, transition notifications, and behavior-changing activations are distinct messages.
5. INIT autonomously requests IDLE only after verified stability. The actual stability criterion
   is deferred. Production defaults to pending, not success; tests supply explicit evidence.
6. Use file-per-node organization and payload-local graph machinery. Do not implement generic
   hierarchical statecharts, parallel regions, or subgraph execution yet.
7. HOLD distinguishes automatic limb waiting from manual hold. Vision may reacquire from limb
   waiting; manual hold requires explicit resume even when valid vision arrives.
8. `GIMBAL_STOW` is a major command requesting system STOW. HOME and GOTO are guarded local
   OPERATE/HOLD self-loop commands. Add local HOLD and RESUME commands.
9. SAFE always inhibits motion. SAFE never homes, slews, or attempts stow. STOW parks the
   mechanism.
10. IDLE/STOW/SAFE disable imaging and inference. INIT only captures for explicitly requested
    verification work. OPERATE preserves current settings by default and supports typed
    overrides.
11. Perform clean public API/schema cutovers. No old-mode aliases, dual-purpose messages, legacy
    telemetry aliases, or public compatibility adapters. Update consumers atomically.
12. Each PR includes its tests and matching as-built documentation. Foundation PRs may introduce
    inactive pure capabilities; there must never be two competing live payload runtimes.

## Authority and data flow

```text
major command / subsystem request -> system-mode authority (teammate)
                                     |
                   transition notification -> storage/downlink/audit only
                                     |
                   authoritative activation -> payload shell boundary
                                                |
                              mode-to-graph mapping, activation-key validation
                                                |
                                   selected pure graph and active node
                                                |
              reference + imaging/inference policy + effects + requests + events
                                                |
                               payload I/O shell / injected runtime services
                                                |
                       rate HAL or detailed-plant inner PI -> torque HAL
```

The system mode does not prescribe payload behavior. Its only payload-facing significance is
selecting a name-mirrored graph at one boundary. Graphs own topology, transition guards, node
behavior, observation acceptance, initialization progress, and imaging/inference policy.

An emitted SAFE or IDLE request never changes the requesting graph. A transition notification,
including an accepted one, never changes payload behavior. Only a validated activation selects
or re-enters a graph. INIT remains INIT after requesting IDLE until IDLE activation arrives.

Containment is a separate authority: fault-owned safety evidence and payload-local hardware
interlocks may immediately inhibit output without changing graph identity. An activation cannot
clear a hardware fault latch, manufacture fresh feedback, or bypass fault recovery.

## Pure versus imperative responsibilities

- Pure graph/node functions receive explicit time, observations, command candidates, and effect
  results. They return new state and typed outcomes. No clocks, logging, hardware, bus, files,
  model construction, futures, or threads live inside a graph.
- `payload/app.py` and small payload-local shell services execute outcomes and own bus/HAL I/O.
  Model loading, hashing files, ONNX construction, and warm-up are shell effects requested by
  INIT.
- `payload/control.py` owns graph-independent servo state and reference execution. It does not
  know SAFE, STOW, TRACKING, HOLD, or `SystemMode`.
- `payload/gimbal/` retains geometry, rate fitting, rate/position primitives, stopping limits,
  integrity checks, and typed references. It contains no graph arbiter or node dispatch.
- `payload/tracking/` retains reusable residual chronology/replay and blob association. Its
  algorithms are graph-independent; residual history and feeding/reset policy belong to OPERATE.
- Preprocessing remains a direct, co-located call chain in `PayloadApp.process_frame()`,
  followed by inference in the same capture path. Do not put a bus or worker boundary between
  them.
- Do not introduce new tensor/mask bus transport. This stack does not broaden or redesign the
  existing `InferenceResultMsg` artifact transport.

## End-state layout

```text
payload/
  app.py                     # bus, capture/control loops, HAL, effect dispatch
  records.py                 # compact explicit observations, IssSample, VisionSample
  state.py                   # shell-facing PayloadState: servo + graph runtime + cadence
  control.py                 # mode-free ServoState / reference execution
  imaging.py                 # imperative camera-policy application and cadence bookkeeping
  lifecycle.py               # bounded effect runner and typed initialization service seams
  graphs/
    __init__.py              # minimal exports; no SystemMode mapping here
    base.py                  # typed specs, edges, policies, inputs, outcomes, effects
    runtime.py               # closed graph-state union and explicit match dispatch
    idle.py                  # one hold node
    safe.py                  # one inhibited node
    stow/
      graph.py  state.py  moving.py  held.py
    init/
      graph.py  state.py  selftest.py  model_load.py  home.py  ready.py
    operate/
      graph.py  state.py  tracking.py  rewind.py  fast_rewind.py  hold.py
  gimbal/                    # reusable pure primitives; arbiter.py removed at cutover
  inference/
    runtime.py               # imperative, lazy model factory/session lifecycle
    ...                      # detector protocols, conditioned contract, ONNX mechanics
  preprocess/                # existing pure transforms
  tracking/                  # existing pure estimator/association primitives
```

The final naming may follow existing class conventions, but not change these ownership
boundaries. Use a minimal explicit mode-to-graph `match` in the shell boundary, not reflection
or a callable registry. A constant enum-to-enum mapping is acceptable data, but keep it outside
pure graphs. Enforce no reverse imports from gimbal/tracking into graphs, and no graph imports
of shell services.

## Typed graph contract

- Define `GraphId` with the five payload-local names and an enum per graph for local nodes.
  Telemetry serializes lowercase paths (`operate`, `fast_rewind`) without weakening internal
  types.
- `GraphSpec[NodeT]`: identity, declared nodes, initial node, immutable directed edges, default
  imaging/inference policy. `Edge[NodeT]`: source, target, trigger, optional typed `CommandId`.
- Explicit triggers cover vision acquire, coast exhaustion, rewind timer, limb arrival, command,
  effect completion, and verified stability. Guards remain explicit code, not callable tables.
- `validate_spec()` returns `Result`; reject missing endpoints, invalid initial nodes, exact
  duplicate edges for any trigger, malformed command edges, and multiple command destinations
  sharing `(source, command_id)`. Ambiguous command matches must fail, not take first.
  Distinct-target automatic edges sharing `(source, trigger)` are legal: guard exclusivity
  belongs to the concrete graph, which selects its directed target edge in code.
- `TickInputs`: explicit monotonic `now_s`, timestamp string when emitting records, encoder and
  navigation observations, optional vision, typed effect results, and command candidate.
- Separate immutable `GraphState` union and typed per-node state. Share genuinely common operate
  target/association/residual state; do not allocate duplicate histories per operate node.
- `NodeOutcome`/`GraphOutcome`: typed reference, concrete effective policies, optional
  transition intent, effect intents, optional system-mode request intent, fault codes, and
  compact events. Graph runtime validates and commits node transitions; nodes cannot assign
  arbitrary node names.
- Keep system-request intent payload-local (for this stack, SAFE and INIT_COMPLETE). Translate
  INIT_COMPLETE to an IDLE bus request in the shell, not via `SystemMode` imports in graph code.
- Node step: `(state, inputs, parameters) -> (new_state, NodeOutcome)`.
  Graph step: `(graph_state, inputs, parameters) -> (new_graph_state, GraphOutcome)`.
  Command application:
  `(graph_state, command, inputs, parameters) -> Result[CommandOutcome, FaultCode]`.
  Graph `initial_state(inputs, parameters)` explicitly captures entry pose/time where needed.
- References form a closed union: `RateReference`, `PoseReference`, `StowReference`,
  `InhibitReference`. They live below graphs, in gimbal reference types. Rate/pose include
  explicit travel envelope values, not graph/mode names. Servo primitives apply finite hardware
  limits and stopping guards.
- A pose reference is evaluated against fresh encoder feedback at servo cadence. IDLE/manual
  HOLD hold a captured pose, not merely a zero rate that permits drift. Missing entry feedback
  inhibits until valid feedback establishes the hold target.
- Production STOW uses the existing bounded `stow_reference_step()` HAL path. Detailed-plant
  STOW uses position-to-rate-to-PI with stow-switch evidence. Do not replace bounded production
  stow with an unbounded proportional slew or allow science limits to prevent reaching the stow
  pose.
- Effects and results carry `(activation_key, effect_id)`. Submit each once; consume each once.
  Late results from old activations never advance a new graph.
- Perform at most one node transition per outer tick. On a transition, emit entry reference and
  effective policy for the destination; do not send another tick of the source's obsolete
  motion. Priority: local containment, activation, valid command, then automatic graph edges.
- Use closed unions, frozen slots dataclasses, enums, typed Protocols, and explicit `match`.
  No general FSM framework, `getattr`, duck typing, plugin registry, or string-based dispatch.

## Graph inventory, edges, and policy defaults

| Graph | Initial behavior | Internal transitions | Imaging/inference |
| --- | --- | --- | --- |
| IDLE | Capture and hold valid entry pose | Single node | Off |
| STOW | Bounded move to stow | moving -> held on confirmed arrival | Off |
| SAFE | Confirm actuator inhibit | Single node; no motion references | Off |
| INIT | Explicit self-test work | selftest -> model_load -> home -> ready | Only explicit test work |
| OPERATE | Cold TRACKING, as today's startup policy | tracking/rewind/fast_rewind/hold | Existing defaults |

OPERATE edges:

- TRACKING -> REWIND on coast exhaustion away from limb.
- TRACKING -> HOLD(limb_wait) on exhausted coast at limb.
- REWIND -> FAST_REWIND on timer; both hunts -> TRACKING on accepted vision.
- Both hunts -> HOLD(limb_wait) on limb arrival without accepted vision.
- HOLD(limb_wait) -> TRACKING on accepted vision. HOLD(manual) never follows that edge.
- TRACKING/REWIND/FAST_REWIND -> HOLD(manual) on `GIMBAL_HOLD`.
- HOLD -> HOLD on `GIMBAL_HOME`, `GIMBAL_GOTO`, or repeated `GIMBAL_HOLD`. HOME/GOTO make the
  hold manual; their result must not remain auto-reacquiring limb wait.
- HOLD -> cold TRACKING on `GIMBAL_RESUME`; reject RESUME from other nodes.
- Reject HOME/GOTO outside HOLD and reject payload commands in non-OPERATE graphs.

Commands arrive as `RoutedCommandMsg`, never raw ingress messages. Validate the dictionary
schema, active graph, directed edge, current source node, guard, finite parameter values, travel
envelope, fresh feedback, and containment before commit. A rejection changes no graph/setpoint
state. Correlate ACK/NACK by existing `(source, seq, command_id)`. ACK means
accepted/committed, not pose arrival. Bound duplicate tracking and never reapply a command or
emit competing execution ACKs.

Major command targets are `"system_modes"`: `GIMBAL_STOW` requests STOW and `EXIT_SAFE` requests
exit to IDLE, retaining EXIT_SAFE ARM/EXECUTE protection and fault eligibility. The teammate
owns these execution ACKs and major edge guards. Do not use `"core"` because the current router
treats all core commands as directly executed. Hazard class is independent of major/minor
ownership.

`ImagingPolicy`: acquisition enabled, capture interval, duty cycle, exposure, gain, product
enable. `InferencePolicy`: enabled, positive integer decimation, typed model selection
(initially current configured pair). Resolve graph defaults and node overrides into a complete
effective policy. Use existing validated sensor bounds; configuration defaults must equal
`config/default.toml`. Keep outer/inner control rates independent of capture rate. Do not alter
bands, tensor geometry, preprocessing normalization, model weights, thresholds, or physical
controller constants here. The current networks require image+GSD conditioning: retain that
contract; no disable-GSD knob.

Disabled acquisition means stop the sensor, not continually expose and discard. While
acquisition is enabled but a duty/cadence gate skips processing, drain buffers without
inventing vision misses. Apply settings before restarting capture; check every HAL `Result`. Do
not repeatedly start/stop or rewrite unchanged settings. Reset phase deterministically on
effective-policy changes. Validate intervals against camera rate and exposure; decimation counts
accepted captures only. Reject stale frame/inference commits across activation or incompatible
policy revisions. Manual HOLD may keep capture/detection running for telemetry but never resumes
itself.

## Clock, ordering, boot, and restart

The existing `flight.libs.time.Clock` already separates monotonic and UTC time. The composition
root creates/injects it (`core/main.py` for flight, `sim.sil` for simulation). Preserve this
design. Graph deadlines use supplied monotonic time; ephemeris uses supplied UTC. Do not order
activations by timestamps or create a new core clock service.

Use an authority-owned activation sequence scoped to a composition-root-provided session epoch.
The root creates the epoch once; deterministic test roots inject a fixed value. Sequence
ownership belongs to the teammate's authority; same-process authority restarts must preserve its
sequence. A whole flight restart gets a new epoch and rebuilt consumers. Payload never creates
an epoch.

No activation means `GraphState | None`, imaging off and actuator inhibited. This is not a sixth
system mode or an implicit OPERATE. On startup/restart payload requests current state over the
bus: the bus does not retain publications for future subscribers. A snapshot response uses the
current activation, not a fabricated transition. Heartbeats remain active while awaiting
synchronization.

Duplicates do not re-enter nodes, reload models, or resend poses. Older sequences are ignored.
Conflicting contents under the same key trigger a reported synchronization error and
containment. Reject an unexpected epoch. Accept self-contained newer activations across
sequence gaps and report the gap. Process queued safety activations in order; never coalesce
away SAFE containment. A same-mode activation with a newer key explicitly re-enters with fresh
graph-local state. The bus is trusted in-process transport, not publisher authentication; do not
claim otherwise.

## Shared message contract (freeze in PR 1, implement at PR 5)

All envelopes retain `msg_type`, `timestamp_utc`, and `schema_version`. New records use frozen
slots dataclasses. Incompatible message changes increment the global schema version.

| Message | Required semantic fields | Only behavioral consumer |
| --- | --- | --- |
| `SystemModeRequestMsg` | request_id, requested_mode, requested_by, reason, optional command correlation, optional originating activation key | Teammate's mode authority |
| `SystemModeTransitionMsg` | transition_id, request_id, epoch, previous_mode, requested_mode, resulting_mode, ACCEPTED/DENIED decision, reason, optional activation_sequence | Audit/storage/downlink; never payload behavior |
| `SystemModeActivatedMsg` | epoch, sequence, previous_mode (optional for initial snapshot), active_mode, reason, request_id (optional), recovery_authorized | Payload activation boundary and other behavioral subscribers |
| `SystemModeSyncRequestMsg` | subscriber, expected_epoch, last_sequence (optional), request_id | Mode authority; replies with current activation |

Keep command correlation in a typed embedded record `(source, seq, command_id)`, not an untyped
dictionary. The shared activation-key record contains only epoch and sequence, not a clock
source. `recovery_authorized` defaults false and can only be true for an authority-approved
EXIT_SAFE. Ordinary activations do not clear fault-owned or hardware-owned latches.

PR 5 places the shared `ActivationKey` in `flight.libs.types.activation`; payload records
consume it but do not own it. `CommandCorrelation` is a frozen embedded message record
with `source`, `seq`, and `command_id`. The incompatible bus schema becomes version 3.
Recovery authorization is consumed only for a newer SAFE-to-IDLE activation with a nonempty
request identity. Fault-owned safety evidence has no system-mode field: it carries
`evidence_epoch`, `evidence_sequence`, `observed_s`, and optional `recovery_request_id`
alongside the latch and active faults. Payload recovery requires matching released evidence,
the current authorized activation, and fresh confirmed inhibited hardware. A request or audit
record cannot release either latch.

There is no requirement that every notification imply activation: denials never do; an authority
may record a committed transition before activation. Correlate records and activation by
identity, not assumed cross-subscription queue order. Reconstruct current behavior from
activation alone. Persist and downlink the dedicated transition record through explicit
consumers in existing storage/downlink services; do not publish a duplicate generic event as
another canonical record.

At cutover remove `ModeChangeMsg`, `MessageType.MODE_CHANGE`, the old system-mode members, and
`GimbalState`. Node enums move into their graphs. `GimbalCommandMsg` replaces `state` with
`payload_graph` and `payload_node`. Pointing/transition telemetry and analysis use graph/node
fields, not a renamed `gimbal_state` field. Remove the deprecated single-target-ID telemetry
while migrating consumers; aggregate plume tracking remains the control policy.

## INIT completion and model lifecycle

INIT issues sequential, idempotent effect intents:

1. Self-test: collect explicit device/service health evidence.
2. Model load: create/verify the configured classifier+segmentor pair and perform warm-up.
3. Home: latch configured home pose and wait for explicit arrival evidence.
4. Ready: wait for `InitVerificationResult` scoped to the current activation.

`InitVerificationResult` has PENDING/VERIFIED/FAILED status and an evidence identifier. The
production provider returns PENDING until the future stable-state criterion is implemented. Do
not infer VERIFIED from elapsed time, absence of a fault, successful model load, or pose
arrival. Do not add a default-true flag, an operator bypass, or an implicit INIT->OPERATE
transition. After VERIFIED emit one correlated IDLE request per activation and remain ready
pending activation. Test denial/no reply/duplicate verification without a request flood;
request retry policy is deferred unless the authority's delivery contract requires one.

Construct an injected lazy inference-runtime service at the composition root, not sessions
there. Its typed factory is configured by the root and executes in the payload shell. Do not
import concrete drivers in graphs/app, and do not construct a separate bus or clock. SDK loading
and file verification remain lazy. Catch supported load failures at the runtime seam and return
`Result`; graph effects never depend on startup exceptions from ONNX constructors. Model
preparation runs in one bounded shell worker, outside control/capture locks. It does not move
preprocessing/inference onto a bus or run concurrent detection against an unpublished pair.
Warm-up uses explicit contract-valid inputs and injected status, not science observations.
Publish a usable backend only after complete verification; reject late completion after exit.
Once initialized, ordinary IDLE/STOW/SAFE transitions retain the usable runtime; re-entered INIT
explicitly reruns verification and may reuse sessions only when the same pair is already
verified.

`core/model_deploy.py` currently stages and changes deployment metadata; it does not actually
replace the payload's constructed detector. Preserve that boundary and avoid claiming otherwise.
INIT prepares the configured pair in this stack. Do not invent hot-swapping uploaded artifacts,
cross-import core deployment services, or report that deployment metadata proves runtime
readiness. Uploaded-pair installation/hot-swap remains separate work unless the teammate already
supplies a typed artifact-selection contract; a contract surprise returns to the director before
changing scope.

## Migration surfaces

Paths are relative to the repository root.

- `packages/flight/src/flight/payload/gimbal/arbiter.py`: `ArbiterState`,
  `GimbalArbiter.step`, SAFE and non-SAFE clearing, blob-age coast, miss count, rewind timer,
  implicit limb wait.
- `packages/flight/src/flight/payload/control.py`: `ControlState` nests arbiter, residual
  history, target, and pose; `ingest_inference` gates/associates blobs; `inner_step` branches on
  SAFE/pose; `outer_step` combines graph transitions, scene choice, residual replay, pose/rate,
  and telemetry.
- `packages/flight/src/flight/payload/gimbal/outer.py`: `outer_rate(mode, ...)`,
  `smear_cap_rad_s`, `clip_rate`, `stopping_cap`, `finish`, and private `_rate_decision`.
- `packages/flight/src/flight/payload/gimbal/scene.py`: mode dispatch in `select_scene` and the
  residual acquire/reset policy in `acquire_resets_residual`.
- `packages/flight/src/flight/payload/app.py`: `poll_mode_changes`, `handle_commands`,
  `capture_this_opportunity`, `process_frame`, `advance_outer`, `advance_inner`, `run`,
  `SafeLatch`, `PoseIntent`, encoder interpolation/consumption, actuator recovery, leased rate
  and torque writes, direct pose bookkeeping, and shared-state merge/locks.
- `packages/flight/src/flight/hal/interfaces/gimbal.py`: production rate versus detailed torque
  protocols; `stow_reference_step` is a safety-bearing existing behavior, not just a pose
  setter.
- `packages/flight/src/flight/hal/drivers_real/gimbal.py`: bounded switch/timeout-referenced
  stow, independent inhibit evidence, feedback time mapping, rate leases, and shutdown.
- `packages/flight/src/flight/hal/interfaces/sensor.py` and real/sim sensor implementations:
  acquire/drain, exposure/gain setters, and start/stop already exist. No SDK imports in pure
  graphs.
- `packages/flight/src/flight/payload/inference/{detector,classifier,segmentor,onnx_session,verify,contract}.py`:
  SDK-lazy modules but eager session construction when concrete backends are instantiated;
  conditioned image/GSD contract; hashing is file I/O despite old descriptive purity language.
- `packages/flight/src/flight/core/select_drivers.py`: constructs current detector and applies
  initial exposure/gain before payload starts. Lifecycle ownership changes here in PR 7.
- `packages/flight/src/flight/core/{composition,main,scheduler}.py`: one clock/bus, bus bounds,
  startup health gate, supervised app lifecycle, and driver-agnostic app wiring.
- `packages/flight/src/flight/fault/{policy,app}.py`: fault-to-SAFE requests, authoritative
  safety latch, and current EXIT_SAFE direct un-latching/ACK behavior.
- `packages/flight/src/flight/libs/types/{enums,__init__}.py`,
  `libs/messages/{messages,__init__}.py`, `libs/commands/dictionary.py`: old enums, dual-purpose
  message, pose telemetry, commands/targets/schema.
- `packages/flight/src/flight/core/{routing,command_router}.py`: dictionary-derived targets,
  hazardous ARM/EXECUTE, special core command execution, and safety-state prechecks.
- `packages/flight/src/flight/core/{storage,downlink}.py`: currently persist/downlink generic
  telemetry, not a dedicated system-mode transition class. Explicit notification transport
  needed.
- `packages/flight/src/flight/libs/config/{config,__init__}.py`, `core/config_loader.py`,
  `config/default.toml`, relevant `profiles/*.toml`: typed settings/default equality.
- `packages/sim/src/sim/sil/{runner,validation,stepping,environment_bind}.py`: thread payload
  state, call legacy SAFE signatures, start in tracking, expose gimbal state, and independently
  perform capture/drain. Both simulation and production must use the same policy application
  seam.
- `packages/gse/src/gse/{harness,orchestrator,scenario}.py`: captures mode requests as if
  active; `mode_is` currently means SAFE-ever versus not-SAFE, not terminal authoritative mode.
  Migrate to actual last activation and explicit initial mode in scenarios. Do not rename this
  proxy.
- `packages/tools/src/tools/analysis/{datapoints,recorder,runner,report}.py` and
  `plots/payload.py`: inspect old flattened controller/arbiter state, gimbal-state columns,
  mode-change message counts, and final-gimbal-state outcomes.
- Mirror tests in `packages/flight/tests/{payload,core,fault,libs,hal}`,
  `packages/sim/tests/sil`, `packages/gse/tests`, `packages/tools/tests/analysis`.
  `packages/flight/tests/conftest.py` also imports old arbiter state.
- Scenario fixtures in `scenarios/`; source/doc mirrors in `docs/{flight,sim,gse,tools}`;
  `.importlinter`, `CLAUDE.md`, docs indexes, and requirement traceability.
- No discovered analysis-study controller imports require an architecture migration. Include a
  fresh reference search at cutover, not an assumption based on an earlier conversation's count.

## Stack shape and review boundaries

```text
main -> PR 1 -> PR 2 -> PR 3 -> PR 4 -> PR 5 -> PR 6 -> PR 7 -> PR 8
                                                  ^
                           authority integration contract coordinated with teammate
```

Each PR bases on its predecessor. Each is buildable and honestly documents its as-built
surface. PRs 2-4 prepare reusable/inactive pure components without a competing live runtime.
PR 5 is the single public bus/state/command/telemetry cutover and intentionally spans packages.
PRs 6-7 add the configurable pipeline and initialization effects to that runtime, not
alternative controllers.

The teammate can implement against the PR 1 frozen interface while foundations are built.
Shared types become code in PR 5. Coordinate their authority patch/PR against that contract; do
not implement it in this stack, cherry-pick ownership blindly, or silently add a production
stand-in. PR 5 may be reviewed/tested with isolated test doubles if the authority is not ready,
but live flight stays unactivated and inhibited. Actual authority wiring and final acceptance in
PR 8 are blocked until their implementation is available. Do not declare the complete
architecture operational based solely on test-injected activations.

### PR 1: Record the architecture and shared authority contract

Branch suffix: `01-design-contract`. Title: `Document payload graph architecture and mode
contracts`.

- Add this document containing the complete enduring architecture, interfaces, graph inventory,
  defaults, authority boundary, behavior changes, migration table, teammate contract, PR
  dependencies, and acceptance matrix.
- Add focused new architecture decisions for graph ownership and
  request/notification/activation separation. Never rewrite an accepted decision body. Update
  indexes and the design-doc inventory.
- Record the deferred stable INIT criterion and the metadata/runtime model-deployment
  distinction. Do not describe proposed behavior as as-built or modify runtime code.
- Obtain/record teammate agreement on target `"system_modes"`, shared message fields,
  sequence/epoch ownership, snapshot response, recovery authorization, command ACK ownership,
  and boot. Agreement is currently pending; the teammate has not yet confirmed.
- Verification: `uv run python scripts/check_docs.py --strict` and
  `uv run python scripts/check_adr.py --strict`.
- Exit: the design document and shared contract are ready for teammate review; teammate
  confirmation remains an external gate before authority integration, not a settled agreement
  claimed by this stack.

### PR 2: Extract mode-free gimbal and reference primitives

Branch suffix: `02-primitives`. Title: `Separate gimbal primitives from mode policy`.

- Change `gimbal/outer.py` into mode-free smear, clip, stopping, boundary, and rate-decision
  helpers. Expose a named public rate-decision helper replacing the private helper when nodes
  need it.
- Split `scene.py` into explicit CoG and boresight prediction entry points with optional
  navigation. Move acquire/reset policy to the graph-policy extraction boundary, not the
  geometry library.
- Add the reference union and explicit travel-envelope types in `gimbal/request.py`. Preserve
  direct pose bookkeeping's typed metadata without making it a bus command.
- Keep the existing runtime until PR 5; make existing controller policy composition explicit at
  its current caller so removal of mode-bearing library APIs is atomic. No wrapper retaining
  `outer_rate(mode, ...)` or `select_scene(mode, ...)`.
- Update affected re-exports and tests/mirrors. Preserve physical equations, SI units,
  chronology, unknown-navigation distinction, elevation-only smear budget, and separate
  production/plant paths.
- Tests: current tracking/rewind/fast rates and limit flags; finite hardware cap;
  bounds/stopping; missing navigation versus true zero; hunt boresight never stored as target;
  pose envelope cases.
- Verification: targeted `test_outer.py`, `test_scene.py`, `test_gimbal_request.py`,
  `test_payload_controller.py`; scoped Ruff/mypy plus import/docs gates where touched.
- Exit: no mode dispatch in shared rate/scene helpers; existing runtime numerical tests
  preserved.

### PR 3: Define typed graph contracts and runtime invariants

Branch suffix: `03-graph-contracts`. Title: `Add typed payload graph contracts`.

- Add `payload/records.py` and `graphs/base.py`; explicit observations, local graph/node enums,
  immutable specs/edges, policies, references, events, effect identities, outcomes, and
  commands.
- Add pure validation and activation-key acceptance helpers. They consume primitive shared
  identity values without importing a system-mode authority or constructing session IDs.
- Define full-spec validation, command-edge matching, one-transition semantics, outcome policy
  resolution, and Result errors. Do not build the five concrete graphs or wire a second
  runtime.
- Add import contracts forbidding shared primitives -> graphs and pure graphs -> payload
  shell/HAL drivers; keep the existing peer-app isolation contracts.
- Tests: missing initial/source/target; duplicate/ambiguous command edges; wrong
  source/direction; self-loop legality; unchanged rejection; complete override resolution;
  invalid cadence/decimation; duplicate/stale/conflicting keys and unexpected epoch; pure
  deterministic outcomes.
- Verification: new `packages/flight/tests/payload/graphs/test_base.py` and related contract
  tests, reference tests, scoped Ruff/mypy, `uv run lint-imports`, strict docs gate.
- Exit: a typed contract can express all planned graphs without Any, reflection, or callable
  dispatch.

### PR 4: Implement the five pure graphs and file-per-node policy

Branch suffix: `04-pure-graphs`. Title: `Implement pure payload mode graphs`.

- Build `graphs/runtime.py` with explicit closed-union dispatch and graph-local entry/exit
  state.
- Implement the five graphs and node files shown in the architecture, reusing PR 2 primitives.
  Extract pure operate composition from current controller, with graph-owned
  target/association/residual state and no SystemMode imports. The old live orchestration is
  removed in PR 5.
- Preserve residual event chronology, delayed replay/dedup, shutter encoder interpolation
  inputs, predictor-reference replacement, aggregate centroid/liveness, missing navigation, and
  bounded coast.
- Make limb HOLD versus manual HOLD explicit. Add exact directed commands/guards from the edge
  list.
- SAFE only emits inhibit. STOW moving/held require typed completion evidence. INIT emits effect
  intents and waits for results; no worker/I/O and no production auto-verified criterion.
- Test each node separately and every edge, boundary, priority, re-entry/reset, failure result,
  HOME/GOTO manual conversion, and suppressed stale effect completion.
- Keep the public runtime unchanged until the atomic cutover. Use existing numerical test cases
  as characterization inputs; do not maintain a permanent second implementation/oracle.
- Verification: new graph tests, existing residual/tracker and relevant gimbal tests, scoped
  Ruff/mypy, import/docs gates. No measurement benchmarks or controller retuning.
- Exit: complete pure graph vocabulary/behavior with testable policy/effect outputs, not yet
  live.

### PR 5: Atomically cut over bus, runtime, commands, safety, and consumers

Branch suffix: `05-runtime-cutover`. Title: `Activate payload graphs through authoritative mode
messages`.

This is the deliberately larger migration PR. Do not split it into public compatibility stages.
Use small reviewable commits within it for contracts, shell integration, and consumer
migration.

- Implement the shared message contracts, canonical SystemMode members, discriminants, exports,
  identity/correlation records, and schema bump. Remove the old message and gimbal-state enum.
  Remove `SafetyStateMsg.mode`: the fault latch is safety evidence, not the active system mode.
- Fault policy and startup-health failure publish requests only. Shells supply unique request
  identities; pure policy never reads clocks or generates randomness. The fault latch still
  inhibits immediately through safety evidence, even if the mode authority denies/delays SAFE.
- Retarget STOW/EXIT_SAFE to `"system_modes"`; add HOLD/RESUME dictionary entries targeting
  payload. Preserve existing parameter validation, per-source ordering, and hazardous EXIT_SAFE
  phases. Remove FaultApp's old direct EXIT_SAFE execution/ACK. Authority owns the mode
  decision/ACK.
- Implement fresh safety-evidence eligibility and explicit recovery-authorization consumption:
  FaultApp only releases its latch on a matching authorized EXIT_SAFE activation with cleared
  conditions. Payload separately checks confirmed inhibited/fresh hardware health before
  clearing its local fault latch. Failure remains contained and reports a fault; ordinary mode
  changes never call `_clear_actuator_fault_for_ground()`.
- Implement one shell activation boundary and boot synchronization. Process requests/records as
  non-behavioral. Install root-provided epoch and authority sequence checks, request a snapshot,
  and wait inhibited/off without an activation. No fake authority in production.
- Introduce `PayloadState` and mode-free servo execution. Replace arbiter/pose shadow state with
  graph-owned state and typed references. Delete `gimbal/arbiter.py`, old arbiter exports, old
  controller SAFE flags, and `PoseIntent`. Replace `SafeLatch` with explicit containment state
  that never selects a graph or computes a stow target.
- The control path owns graph state and actuation. Reuse/refactor the existing detailed inner
  worker into an independent control worker for both rate and torque HALs, so slow
  capture/detect/model work cannot prevent processing containment and activations. Graph outer
  cadence and detailed inner cadence remain explicit; production never starts a torque PI.
- Preserve all feedback freshness/time-mapping checks, hardware/science envelope distinctions,
  lease deadlines, recovery bounds, encoder sample IDs, historical sample consumption, catch-up
  caps, integrity detection, bounded stow completion, and confirmed-inhibit shutdown. An
  activation cancels obsolete references and pending effects, flushes vision, resets graph-local
  state, and resets PI/reference memory without discarding valid physical encoder history.
- Use short state locks, never held across camera acquisition, detection, ONNX load, or HAL
  calls. Serialize actuation/activation processing and validate the current activation token
  before publishing an output. No stale outer/inner snapshot may restore old graph state or
  motion.
- Commands are committed on the control path under graph guards, not latched as pending poses in
  the capture loop. Rejected commands leave graph/setpoint state unchanged; preserve ACK
  correlation.
- Wire conservative capture/inference enable defaults now: stop acquisition in disabled graphs,
  run the current default pipeline in OPERATE, and leave INIT awaiting unimplemented shell
  effects. Full override/cadence configuration lands in PR 6; deferred effects are not
  fabricated successes.
- Persist/downlink dedicated transition notifications. Requests, notifications, and activations
  get explicit queue policies; activation/request/sync and safety evidence are NEVER_DROP.
  Notification backlog is observable and its audit transport is non-behavioral.
- Migrate `TickOutcome`, `GimbalCommandMsg`, pointing/transition events, simulation state
  threading/accessors, GSE capture/scoring, analysis columns/plots/report outcomes, all affected
  fixtures, root instructions, and source/doc mirrors atomically. Use `payload.graph`,
  `payload.node`, and activation epoch/sequence, with inactive graph-local estimates reported
  unavailable, not zero.
- GSE `mode_is` compares the last actual activation; no activation is unknown, not IDLE/nominal.
  Update scenario setup to explicitly activate the intended starting mode in test fixtures.
  Rename report outcomes to final graph/node/system mode; no legacy columns retained.
- Rename `ArbiterConfig` and `controller.arbiter` to `OperateGraphConfig` and
  `controller.operate`; migrate exports, TOML loader/defaults, affected profile overrides/tests.
  Preserve parameter values. Do not rename unrelated residual-estimator compatibility fields.

Tests and exit criteria:

- A request or notification, whether accepted/denied, cannot change payload graph or release
  motion.
- All five activation mappings; no activation boot; late-subscriber sync;
  duplicate/stale/gap/conflicting-key/wrong-epoch behavior; same-mode new-key re-entry; SAFE
  cannot be skipped.
- SAFE calls inhibit, never stow/home/goto/rate/torque. Fault containment works while OPERATE
  remains the selected graph and while detection is blocked. Healthy activation does not clear
  fault latch.
- Authorized recovery succeeds only with explicit evidence; rejected EXIT_SAFE never un-latches.
- Command wrong graph/node/direction, bad/nonfinite params, stale feedback, travel bounds,
  duplicate execution, and competing activation produce a single correlated result without
  obsolete motion.
- Rate and torque fake HALs; existing real-adapter fake transport; stow switch/timeout; no
  science-envelope leakage into stow; no production torque command; stale leases still inhibit.
- Storage/downlink notification schema/priority/identity; GSE terminal-mode scoring with denied
  requests and multiple activations; every migrated extractor returns intentional values rather
  than exceptions hidden by the recorder's NaN/empty-string fallback.
- Verification: affected flight libs/core/fault/payload tests, real/sim actuator regression
  modules, affected SIL/GSE/analysis-recorder tests; workspace mypy/Ruff and
  import/schema/docs/VCRM checks. Run narrowed test selectors; do not run every suite
  repeatedly during worker rework.
- Exit: one live graph runtime, no forbidden legacy APIs/fields, and a clean public schema
  cutover. Tests may inject authority messages, but mark actual authority integration as pending
  until PR 8.

### PR 6: Execute graph-owned camera and inference policies

Branch suffix: `06-imaging-policy`. Title: `Apply graph-scoped imaging and inference policies`.

- Add `payload/imaging.py` and typed payload policy config records. Graph defaults and node
  overrides resolve complete policy; shell tracks applied policy revision, streaming state, and
  cadence only.
- Centralize policy application so `run()` and SIL `step_once()` do not independently implement
  capture/drain behavior. Use one payload shell method for capture opportunities and enable
  gates.
- Apply exposure/gain while stopped, then start only when enabled. Stop when disabled. Keep
  off-duty buffer drains only for a running stream; handle setting/start/stop/acquire errors via
  Result/fault containment, never as a false successful policy application.
- Schedule capture interval and duty independently from controller cadence; reset phase on
  policy revision. Inference decimation counts successful captures; non-inferred frames are not
  plume-loss observations. Capture-only operation is allowed; no detector call or fabricated
  empty result.
- Capture context includes activation key, policy revision, and model identity. Check context
  before committing vision/products. Discard superseded completions and never feed them to a
  re-entered graph. Treat products from superseded work consistently as discarded by default.
- Keep ProcessedFrame local and preprocessing immediately followed by any requested inference.
  Preserve live exposure metadata in smear calculations and mandatory GSD/model shape contract.
- Support graph/node exposure, gain, cadence, inference cadence, and product overrides using
  validated typed settings. Default OPERATE values remain unchanged.
- Update sensor/default config documentation, policy telemetry, SIL helpers, relevant capture
  driver stand-ins/spies, and analysis availability semantics.
- Tests: off graphs call neither acquire nor detector; stop/start and unchanged-policy
  idempotence; actual exposure/gain overrides; invalid intervals/settings; duty/inference
  decimation boundaries; no inference != empty vision; failure injection for each HAL method;
  activation during blocked capture/detect; superseded product suppression; equal deterministic
  policies in SIL and app.
- Verification: payload app/imaging tests, preprocess/inference tests touched, sensor
  fake-driver tests, config-default/profile tests, SIL imaging-policy tests; scoped Ruff/mypy,
  import/docs gates.
- Exit: graph policy controls real HAL methods and the deterministic simulation through one
  seam.

### PR 7: Execute INIT lifecycle effects and deferred verification

Branch suffix: `07-init-lifecycle`. Title: `Run INIT through explicit verified lifecycle
effects`.

- Add `payload/lifecycle.py` and `inference/runtime.py` Protocol-typed
  services/factories/effect results. Root selects configuration/backend factory; shell decides
  when INIT loads/warm-ups.
- Change `Drivers`/`select_drivers`/`build_apps` and corresponding SIL construction to inject
  lazy inference runtime, not eagerly constructed real sessions.
- Move graph-specific initial exposure/gain decisions out of driver selection into policy
  application, retaining safe constructor constraints and SDK-lazy imports.
- One bounded effect worker executes model load/hash/contract checks/warm-up. The control worker
  never blocks on it. Home/self-test actions return typed evidence; graphs own progression.
  Integrate shutdown/cancellation without waiting indefinitely for SDK work.
- Implement the explicit initialization verifier seam, production PENDING implementation, and
  deterministic test implementation. No configurable pass-by-default escape hatch.
- INIT ready emits one IDLE request after current-token VERIFIED and all prerequisite results.
  Wait for actual activation; authority denial/no reply leaves INIT ready/inhibited as
  configured.
- Keep usable sessions after leaving INIT; reject publishing a late backend after graph exit. A
  failure preserves any previously verified usable runtime while INIT remains failed/pending and
  emits appropriate fault evidence. Never silently enable OPERATE with an unverified backend.
- Keep model deployment's current metadata responsibility separate; document real runtime model
  identity distinctly. Do not implement uploaded-model hot-swap as an unreviewed side project.
- Tests: ordered successful effect sequence; each failure; missing verifier stays INIT; home
  needs valid arrival evidence; duplicate/old-token results; repeated ready ticks issue one
  request; blocked load does not block SAFE inhibit; cancelled effect cannot install sessions;
  re-entry reruns verification; imports need no SDK; factory/load/warm-up failures return
  Result. Successful completion uses injected evidence, never invented stability thresholds.
- Verification: lifecycle/init graph tests, inference factory/session/verify tests, core
  driver/composition tests, SIL model-upload regression for unchanged deployment boundary,
  focused lazy-import/flight-image checks, scoped Ruff/mypy and docs/import gates.
- Exit: INIT lifecycle is executable and waits honestly for the deferred verification
  criterion.

### PR 8: Integrate the teammate's authority and close end-to-end acceptance

Branch suffix: `08-integration-acceptance`. Title: `Validate mode-driven payload graphs end to
end`.

Prerequisite: actual teammate-owned authority implementation using the frozen shared contracts.
If unavailable, report this blocker and leave the stack ready through PR 7; do not ship a
stand-in.

- Integrate actual authority through composition, scheduler, startup snapshot, heartbeat
  monitoring, SIL step ordering, and GSE/test scenario setup. Wire via bus; payload never
  imports the authority. Do not rewrite the teammate's major-mode graph or INIT stability
  criterion.
- Add actual request -> arbitration -> notification -> activation integration tests for major
  STOW/EXIT_SAFE, fault SAFE, INIT's verified IDLE request, grants, denials, and delayed
  activation.
- Expose distinct observables for requested mode, recorded transition, authoritative active
  mode, payload graph/node, containment state, and runtime model readiness/identity.
- Update representative existing scenarios and add graph/command/INIT scenarios with explicit
  initial mode. Preserve scenario assertions' physical intent; do not weaken inference/motion
  assertions merely to accommodate imaging now correctly being off in IDLE.
- Require explicit test verifier injection for nominal INIT->IDLE scenarios. Production still
  waits for the future verifier. Tests demonstrate the seam, not flight-ready stable-state
  proof.
- Finish as-built source/doc mirrors, architecture inventory and mapping, requirements/test
  links, root coding context, migration notes/schema version, and the enduring design-stack
  record.
- Search source/tests/docs/config/scripts for removed enum members, message types, arbiter
  imports, flags, and telemetry fields. Historical decision records may describe old behavior;
  do not rewrite their accepted bodies. Remove obsolete live tests/docs only as part of the
  reviewed source migration.
- Tests: nominal initialization/pending verification; all graphs; denied request has no behavior
  change; SAFE during blocked compute; stale activation/effect/frame; manual hold resists
  vision; explicit resume; major STOW versus local HOME/GOTO; contained hardware cannot be
  recovered by ordinary activation; sequence/restart sync; terminal mode capture; sensor-policy
  parity.
- Verification: focused end-to-end regression modules first, then one full final gate pass
  (Ruff check/format, workspace mypy, lint-imports, VCRM, docs and decision-record strict
  checks, flight-image check, `uv run pytest -m "not e2e"`). No real hardware commands or
  external station use during verification.
- Exit: required CI `gates` evidence and complete local evidence are recorded or limitations
  named; actual authority is integrated; no legacy runtime remains. Deferred stable-state proof
  and uploaded-pair hot-swap are explicitly not claimed complete.

## Sequential handoff protocol

The director owns design, authority decisions, user updates, full diff review, and PR
publication. Implementation and mechanical verification are delegated to a persistent worker
through a scoped handoff for one PR at a time. No concurrent PR implementations or
agent-specific competing branches.

Before each dispatch:

1. Read this architecture memory and the predecessor's accepted handoff report. Check current
   worktree and base SHA. Settle scope/contract surprises personally before handing off edits.
2. Freeze exact interfaces, touched files, tests/assertions, and commands for this PR. The
   worker does not re-investigate settled conclusions or pick alternative designs.
3. Supply a self-contained brief covering: PR number/title; predecessor branch and accepted
   base SHA; goal and non-goals; the exact plan section and this document; confirmed decisions
   relevant to the PR; exact interface signatures/data shape and hard edge-case precedence;
   allowed files plus consumers that must migrate and changes not to touch; exact regression
   cases, assertions/stubs, and narrow verification commands; the no-legacy-alias, no-peer-app
   import, no generic FSM engine, no invented verification rules; runtime state; artifact
   retention; and the prohibition on pushing, publishing, merging, rewriting history, or
   destructive/network actions.

After dispatch:

1. Wait for the report rather than redoing implementation or running the same tests.
2. Personally review the complete diff and evidence at each landing point, including tests,
   configuration/default equality, consumer migration, lock boundaries, safety, and doc claims.
3. Batch all defects into one rework brief for the same PR. Do not accept prose-only "tests
   passed".
4. Once accepted, perform lead-owned commit/branch publication and PR creation after
   appropriate network authorization. The PR targets its predecessor and includes a stack
   index, scope, behavior/API changes, test evidence, deferred work, and predecessor
   dependency.
5. Write a compact execution receipt in this document: branch/commit/base, PR URL (only after
   actual creation), files/interfaces delivered, exact test commands/results, preserved
   jobs/artifact paths, director review verdict, and remaining blockers.
6. Update the user and any promised PR title/description immediately before the next handoff.
7. Start the next PR only after the predecessor's diff is accepted. Build the whole stack
   without repeated user approval pauses, but never merge or force-push/rewrite branches
   automatically.

If predecessors change during review, preserve the stack with ordinary forward merges unless
the user specifically approves history rewriting. Inspect external changes rather than
reverting.

## Verification strategy

Every PR must compile/typecheck its actual dependency surface and pass its focused regression
tests. Use `uv run`, never system Python. Tests mirror source and have no `__init__.py`. Update
a module's documentation in the same PR, including moved/deleted-module pages and indexes. Run
scoped Ruff/mypy during iteration; use workspace mypy at the atomic cross-package cutover. Each
PR's required CI gates remain mandatory; local narrow checks do not waive them.

At the final gate, execute once and retain raw output for:

```text
uv run ruff check packages scripts
uv run ruff format --check packages scripts
uv run mypy packages scripts
uv run lint-imports
uv run python scripts/check_vcrm.py
uv run python scripts/check_docs.py --strict
uv run python scripts/check_adr.py --strict
uv run python scripts/check_flight_image.py
uv run pytest -m "not e2e"
```

After a final-gate failure, rerun affected checks plus direct dependents; do not regate
unchanged work on every small handoff. PR CI excludes slow tests; the final local pass includes
them. Use fake hardware/SDK services and deterministic clocks. Existing loopback-link scenarios
need separate network permission; they are not a reason to run production `flight.core.main`.
Do not add performance benchmarks, retune controller parameters, or generate new model artifacts
to prove an architecture refactor.

## Risks, non-goals, and definition of done

- The main risk is the atomic cutover's concurrency and cross-package schema blast radius.
  Mitigation: tested pure foundations first, full cutover diff review, blocked-compute safety
  tests, strict typing, and complete consumer migration in one PR.
- Teammate authority delivery is an external dependency and is currently unconfirmed. No
  production activation fallback is acceptable. Actual integrated authority evidence is
  required to close the stack.
- The INIT stability criterion remains intentionally unknown. The stack is done when the
  verification seam and autonomous verified-IDLE request are implemented, not when flight-ready
  readiness certification is invented. This limitation must remain visible in docs/PRs.
- SAFE-to-inhibit and manual HOLD are intentional behavior changes, not numerical regressions.
  Keep the existing geometric/control/filter algorithms and parameter values unchanged.
- Do not implement subgraphs, generic reusable FSM machinery, new system modes, an authority
  app owned by this stack, a clock redesign, estimator replacement, new ML
  training/conditioning, uploaded-model hot-swap, or broad artifact-transport cleanup.
- The final implementation has exactly one activation-driven payload runtime; every system mode
  maps to a correspondingly named graph; commands require declared directed guarded edges;
  imaging/inference are graph-owned; shared primitives are mode-free; SAFE inhibits
  independently; INIT waits for real verification; consumers and documentation use the new
  contracts.
- No merge, release, hardware validation, deployment, or PR lifecycle state is inferred from a
  local branch or tests. Report only evidence actually obtained.

## Execution receipts

Latest publication receipt: PR 7 is OPEN as
https://github.com/Hybrinter/Plume-Autonomous-Classification-and-Tracking/pull/120,
stacked on PR 6 at `a0c519b`. The published code/test head is `d10a74d`, containing
implementation `d0bd54e`, ancestry merge `bb368b3`, and the final acceptance receipt.
All workspace static gates pass. Full non-e2e tests: 1445 passed, 8 skipped,
1 known vendor-checkout CRLF provenance failure; vendor source is unchanged.
Initial CI on `d10a74d` is in progress; no CI success is claimed in this receipt.
PR 8 awaits authority-contract alignment and the recovery-path decision:
the actual authority PR 112 uses `EXIT_SAFE -> INIT`, while this stack currently
authorizes recovery through `SAFE -> IDLE`. Neither path is changed here.
The local/unpublished statuses in earlier checkpoint receipts are historical.

PR 4 publication: director review accepted; implementation commit `63bde2b`, branch
`devin/payload-graphs-04-pure-graphs`, predecessor `7d42c1b`.
Published PR: https://github.com/Hybrinter/Plume-Autonomous-Classification-and-Tracking/pull/111 .
The table's readiness entry records the pre-publication gate; this receipt confirms publication.
PR 5 is next. Actual authority integration and teammate contract confirmation remain pending.

PR 5 work has started locally on `devin/payload-graphs-05-runtime-cutover`, based on
`481fdf6`. Runtime and schema implementation is in progress and is not yet verified or
published. Consumer migration separates accepted system mode, selected payload graph/node,
and containment. GSE terminal-mode scoring compares only the accepted activation; absent
activation is unknown. Analysis uses unavailable values for inactive graph estimates.
Explicit scenario activation remains test setup, not a substitute production authority.

PR 5 review checkpoint: the first runtime draft was rejected; the current rework is still
unaccepted and uncommitted. Known remaining work includes atomic capture-context commit
(including activation during acquisition, detection, and product storage), blocked-compute
regressions, HAL-spy no-motion proofs, recovery replay/evidence tests, complete old-controller
regression migration, pointing telemetry assertions, and flight/sim source mirrors.
Focused payload-app verification still has five failures. The strict hardware-angle freshness
guard rejects noisy simulated feedback near the lower stop; no new noise tolerance or
controller tuning has been approved or applied. Preserve the existing physical and numerical
contracts while investigating this boundary instead of relaxing the guard to pass tests.
Lead-owned terminal-mode, extractor-availability, and elevation-block checks pass:
31 tests, raw checkpoint evidence
`C:/Users/kampw/AppData/Local/Temp/pact-pr5-lead-critical-checkpoint.log`.
This is not a full PR 5 verification result. No background verification jobs remain running.

PR 5 boundary rework: the focused payload-app and graph-parameter selector now passes
29 tests. Runtime test fixtures use zero simulated encoder noise at the exact lower stop;
production parameters, simulator behavior, and the hardware-bound freshness guard are unchanged.
An explicit regression rejects a noise-scale excursion below that bound. Scoped Ruff check,
format check, and mypy passed. Raw evidence:
`C:/Users/kampw/AppData/Local/Temp/pact-pr5-boundary-pytest.log`,
`C:/Users/kampw/AppData/Local/Temp/pact-pr5-boundary-ruff.log`, and
`C:/Users/kampw/AppData/Local/Temp/pact-pr5-boundary-mypy.log`.
This scoped result does not accept the complete cutover. Runtime review identified capture
identity being assigned after acquisition and a possible command-plus-automatic double edge;
atomic capture publication, containment tokens, recovery, and actuator regressions are under
active rework. PR 5 remains local, uncommitted, and unpublished.

PR 5 chronology review: a proposed simulator timestamp change was rejected and
`hal/drivers_sim/gimbal.py` was restored to its predecessor contents. The earlier
562-test aggregate passed with that rejected change and is not a current cutover gate.
The SIL composition root now advances the existing injected clock through control
ticks; final SIL and safety verification for this rework is pending. Recovery must
retain the nonfuture feedback requirement without a new tolerance. The analysis
recorder no longer advances that clock a second time and preserves shifted origins.
Lead-owned extractor, recorder/twin, and elevation-block checks pass: 35 tests,
`C:/Users/kampw/AppData/Local/Temp/pact-pr5-lead-critical-chronology.log`.
This evidence does not complete runtime, documentation, or workspace verification.

PR 5 docs-final/runtime-safety checkpoint: still a local, uncommitted, unpublished
draft on `devin/payload-graphs-05-runtime-cutover`. This phase added immediate
containment for any `stow()`/`goto_angle()` reference-metadata HAL failure via the
internal `ReferenceCommit` record (rejected-ACK rollback for routed commands, no
committed-edge events or audit), finite-only encoder validation before history
mutation (`GIMBAL_ENCODER_INVALID` on nonfinite angle or timestamp), and a
payload-local `FaultEventMsg` subscription so an in-context detector
`Err(INFERENCE_NAN)` latches containment on the next control poll while stale
detector errors drop silently. Storage and downlink now have transition-ledger
tests covering accepted and denied `SystemModeTransitionMsg` JSON. The
descriptive flight/sim/GSE/root documentation mirrors were completed for the
activation/schema-3 architecture. Evidence: 146 scoped chronology tests plus
8 post-move spot checks (`pact-pr5-chronology-*.log`), narrow docs-final
selector 96 passed (`pact-pr5-docs-final-review-pytest.log`), scoped
Ruff/format clean on non-lead files, scoped mypy 12 files clean, 20 import
contracts kept, `check_docs --strict` and `check_adr --strict` ok
(`pact-pr5-docs-final-review-*.log`). Lead-owned evidence remains frozen:
35 critical chronology tests and 18 GSE final tests. Full workspace gates are
not run; INIT verification stays undefined; teammate contract blockers and
external authority integration remain open. No hardware validation, CI
acceptance, or publication is claimed.

Final review fixes added batch-drained local-fault polling (no HAL inside the
subscription drain), atomic detector-Err check-and-publish under `state_lock`
(stale work returns a `fault=None` outcome), and immediate deliberate
`_inhibit_motion` on `InhibitReference` commit. First full-gate evidence:
ruff check/format, mypy (504 files), lint-imports (20 kept), check_vcrm,
check_docs --strict, check_adr --strict, and check_flight_image all pass;
`pytest -m "not e2e" -n2 --timeout=180` reports 1317 passed / 8 skipped / 6 failed
(`pact-pr5-final-pytest.log`): a preexisting vendored-Xeryon hash mismatch
from CRLF checkout (file unmodified in git) and 5 lead-owned
tools.analysis slow scenario failures (4 xdist worker crashes plus
`safe_latched` not reaching 1.0 in the injected-fault runs), reported for
lead review rather than fixed here.

PR 5 failure follow-up: all 24 slow runner and statistics tests pass with
`uv run pytest packages/tools/tests/analysis/test_analysis_runner_slow.py
packages/tools/tests/analysis/test_analysis_stats.py -n0 --timeout=0`.
Raw evidence: `C:/Users/kampw/AppData/Local/Temp/pact-pr5-slow-followup-pytest.log`.
The runaway scenario had frozen feedback before initial acquisition because
lower-stop noise suppressed imaging. Its scenario-only encoder-noise setting
is now zero; freeze timing, production defaults, hardware guards, and controller
equations are unchanged. It requests SAFE and latches containment without
changing accepted OPERATE mode. Lead-authored diagnostic evidence is retained
as `pact-pr5-lead-slow-diagnosis.json` and `pact-pr5-lead-runaway-fixed.json`.
The earlier worker-crash mechanism remains unconfirmed; serial execution
avoids concurrent copies of the shared scenario fixture and the added 180-second
timeout. Scoped Ruff, mypy, and strict docs pass after the scenario fix.

The remaining local gate blocker is vendored `Xeryon.py` checkout line endings.
The tracked LF bytes match the pinned SHA-256
`bff3338ecff97c2cb01c18a3d07dafdd1b86a19f2e5ce2a8ac8c21a9825bce77`;
the unmodified CRLF worktree matches those bytes after newline normalization.
The pin, vendor source, Git configuration, and security checks remain unchanged.
User permission to normalize only the local vendor checkout is pending.
PR 5 remains uncommitted and unpublished; no background jobs are running.

PR 5 publication authorization: the user requested committing, pushing, and opening
the cutover PR before continuing PR 6. The vendor checkout stays unchanged; its
local CRLF-only provenance-test failure will be disclosed rather than treated as
a source change or a passing test. Publication does not imply CI acceptance,
authority integration, hardware validation, or completion of PRs 6-8.

| PR | Branch | Base | Status | Evidence | Blockers |
| --- | --- | --- | --- | --- | --- |
| 1 | `devin/payload-graphs-01-design-contract` | `main` @ `37aa8f8` | accepted and published; commit `031f9ed` | PR https://github.com/Hybrinter/Plume-Autonomous-Classification-and-Tracking/pull/108; `check_docs.py --strict` ok and `check_adr.py --strict` ok after preserving stale ignored source caches outside the source tree; raw logs retained | teammate interface agreement pending (external gate before authority integration) |
| 2 | `devin/payload-graphs-02-primitives` | `devin/payload-graphs-01-design-contract` @ `031f9ed` | accepted and published; commit `d1b5bcb` | PR https://github.com/Hybrinter/Plume-Autonomous-Classification-and-Tracking/pull/109; focused controller/gimbal/app/encoder selector: 86 passed; migrated analysis consumer: 12 passed; post-review reference/analysis tests: 20 passed; scoped Ruff and mypy passed; import contracts: 18 kept; strict docs and decision-record checks passed. Raw final and initial logs retained separately under `/tmp/pact-pr2-*.log`. | none |
| 3 | `devin/payload-graphs-03-graph-contracts` | `devin/payload-graphs-02-primitives` @ `d1b5bcb` | accepted and published; commit `7d42c1b` | PR https://github.com/Hybrinter/Plume-Autonomous-Classification-and-Tracking/pull/110; records relocated; typed graph/policy/effect/activation contracts added; HOLD/RESUME declarations remain explicitly unsupported by the current app. Automatic edges permit guarded alternate targets; ambiguous command targets and exact duplicates reject. Initial selector: 124 passed; post-review graph/records/app selector: 60 passed; scoped Ruff/mypy, 20 import contracts, and strict docs passed. Raw final evidence retained under `/tmp/pact-pr3-rework-*.log`. | teammate authority agreement remains pending before integration |
| 4 | `devin/payload-graphs-04-pure-graphs` | `devin/payload-graphs-03-graph-contracts` @ `7d42c1b` | accepted and published; PR #111 remains open | All five pure graphs and closed-union dispatch delivered; no live shell wiring. INIT waits for fresh HOME feedback and later explicit verification; matching failure takes precedence. OPERATE retains rate/replay diagnostics, rejects flagged-vision commands, and preserves activation-scoped deduplication across RESUME. Initial graph/residual/tracker/outer/scene selector: 151 passed; post-edit graph selector: 116 passed; final graph selector: 123 passed. Scoped Ruff check/format and strict mypy (37 files), 20 import contracts, and strict docs passed. Raw historical logs: `/tmp/pact-pr4-*.log`, `/tmp/pact-pr4-rework-*.log`, `/tmp/pact-pr4-final-ruff-check.log`, `/tmp/pact-pr4-final-check-docs.log`. Actual final Windows evidence: `C:/Users/kampw/AppData/Local/Temp/pact-pr4-post-edit-*.log` and `C:/Users/kampw/AppData/Local/Temp/pact-pr4-final-review-*.log`; earlier final evidence also retained as `pact-pr4-final-ruff-check.log` and `pact-pr4-final-check-docs.log` in that directory. | teammate authority agreement remains pending before integration |
| 5 | `devin/payload-graphs-05-runtime-cutover` | `devin/payload-graphs-04-pure-graphs` @ `481fdf6` | published as PR https://github.com/Hybrinter/Plume-Autonomous-Classification-and-Tracking/pull/118 at commit `b4762c3`; local-draft receipts above are historical. CI run 37090163917 failed once: the newer PR4 head carried teammate commits `7149c8f`/`4b491f1`, so the CI merge tree contained a delayed-plume regression test still reading `controller.arbiter`; the local PR4 base was integrated and the reference now reads `controller.operate`. Refreshed CI run 37091272764 reported gates SUCCESS at `eae3f9d` | Activation-driven shell, schema 3 contracts, graph-independent containment, atomic capture publication, metadata-failure rollback, local detector-fault containment, transition audit consumers, and documentation mirrors reviewed. All workspace static gates pass (`pact-pr5-final-*.log`). Full tests: 1317 passed, 8 skipped, 6 failures; affected slow follow-up: all 24 passed serially (`pact-pr5-slow-followup-*.log`). Focused evidence includes 146 chronology, 96 docs-final, 35 lead-critical, and 18 GSE tests; see checkpoint receipts above for exact scope and caveats. | Local vendor CRLF normalization permission pending; authority integration and teammate agreement remain external dependencies; INIT verification criterion undefined |
| 6 | `devin/payload-graphs-06-imaging-policy` | `devin/payload-graphs-05-runtime-cutover` @ `eae3f9d` | published as PR https://github.com/Hybrinter/Plume-Autonomous-Classification-and-Tracking/pull/119 (open); head `a0c519b` carries publication commit `0f5ba63` plus two remote commits (`4b2c351` capture-loop deadline sleep, `a0c519b` `publish_products` in applied imaging-policy telemetry); latest known CI gates SUCCESS | Pure `payload/imaging.py` (`plan_capture` deadline/duty schedule with off-policy WAIT, `record_capture` decimation), typed `PayloadPolicyConfig`/`PayloadPolicyOverrideConfig` (frozen slots) with empty TOML inheritance tables, `operating_policy` graph-then-node resolution with startup `ValueError` validation, `runtime.entry_policy` (OPERATE enters on TRACKING's override), `CaptureShell` schedule plus `pending_fault` control-owned containment, context-gated `_imaging_fault` (stale acquire/drain errors drop; global camera-control failures always report), one context-current `imaging_policy` telemetry seam per applied revision including confirmed off policies, settings stop-apply-start ordering, nonfinite-time rejection before HAL, and documentation mirrors. Evidence: initial focused selector 280 passed (`pact-pr6-*.log`); post-rework selector 166 passed (`pact-pr6-rework-*.log`); workspace Ruff check/format, strict mypy, 20 import contracts, VCRM, strict docs, ADR, and flight-image checks all pass. Fast CI-equivalent combined run (`-m "not slow and not e2e" -n2`): 1343 passed, 8 skipped, 2 failures. One failure was a field-order regression (`test_pactconfig_has_drivers_field_last` requires `drivers` final in `PactConfig`); `payload_policy` was moved before `drivers` and the 82-test config/imaging follow-up passed (field-order contract evidence: `pact-pr6-field-order-followup.log`). The remaining failure is the unchanged vendor CRLF provenance hash artifact (`test_xeryon_vendor`), a known local line-ending environment issue, not a code regression. The slow suite was not rerun; no final full-suite green is claimed. | Review, commit, and PR remain with the lead |
| 7 | `devin/payload-graphs-07-init-lifecycle` | `devin/payload-graphs-06-imaging-policy` @ `a0c519b` | lead-reviewed and accepted; committed locally as implementation commit `d0bd54e` with PR6 ancestry merged in `bb368b3`; not published, no PR opened | Lazy `inference/runtime.py` (`RuntimeSession`/`RuntimeFactory` protocols, `OnnxRuntimeFactory`/`OnnxRuntimeSession`, `ScriptedRuntimeFactory`/`ScriptedRuntimeSession`, lock-protected `InferenceRuntime` with `install_verified`), `Detector.warm_up` exercising both models unconditionally, executable `payload/lifecycle.py` (one bounded lazy worker, token-keyed submit/poll/cancel/shutdown, observed self-test, exact fresh-later HOME arrival, pending-by-default verifier), `PayloadApp` integration (effect submission after reference commit, per-tick poll into INIT `runtime.step`, cancel on reentry/containment/shutdown, fail-closed OPERATE gate for unverified runtime, `model_version` stamped from session identity), `Drivers.inference` with lazy real compute and no startup camera setters, and documentation mirrors (`lifecycle.md`, `inference/runtime.md`, app/inference/composition/select_drivers updates). Evidence: initial focused selector 219 passed (`pact-pr7-pytest.log`), subsequently reviewed and reworked (not accepted as-is). Post-rework focused selector 243 passed on lifecycle, runtime, INIT graph, imaging, payload app/safety/encoder, composition, driver selection, core main, SIL model upload, and SIL validation harness, run at `-n2` (CI parallelism; `pact-pr7-rework-pytest.log`); scoped Ruff check/format, strict mypy on 12 changed sources/tests, strict docs, 20 import contracts, and flight-image all pass (`pact-pr7-rework-*.log`). Environment note: local `-n auto` (~16 workers) repeatedly starved the unchanged real-time `run()`-loop test `test_run_drains_camera_on_skipped_opportunities` -- the control thread lag exceeded the outer catch-up cap, containment latched, and capture stayed off, so the test's completion condition was unreachable; `-n2` and serial runs pass, and the mechanism/test predate this change. Final workspace gate on the committed tree: Ruff check/format (511 files), strict mypy (510 files), 20 import contracts, VCRM, strict docs, strict ADR, and flight-image all pass; `uv run pytest -m "not e2e" -n 2` gives 1445 passed, 8 skipped, 1 failure -- the unchanged vendor CRLF provenance hash artifact `test_xeryon_vendor`, a known local line-ending environment issue, not a code regression (`pact-pr7-final-*.log`). | Commit and PR remain with the lead; teammate authority and the production initialization verifier stay external. PR8 integration dependency (factual, from the grounded PR112 investigation): the authority work is OPEN on `devin/1790974527-system-mode-authority` but still uses the older activation-message `key=ActivationKey` rather than the current epoch/sequence fields, routes recovery `EXIT_SAFE -> INIT` while the accepted payload/fault recovery path is `SAFE -> IDLE`, and retains legacy compatibility; authority contract and recovery alignment are required before it can be integrated. No authority implementation changed in this PR |
