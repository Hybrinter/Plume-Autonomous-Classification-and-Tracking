# FSW Code-Complete + SIL Exercise Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline) or
> superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. TDD per task;
> per-phase gates; commit per phase.

**Goal:** Implement the remaining code-only flight-software capabilities from
`docs/superpowers/specs/2026-06-09-pact-flight-final-state-design.md` behind the existing
architecture, and exercise each end-to-end in the deterministic SIL / declarative GSE scenarios.

**Architecture:** Subsystem-app + typed bus; pure cores; HAL Protocols + lazy SDK; Result[T,E];
composition-root ownership; `build_apps`/`Scheduler`/`step_once` stay driver-agnostic; `gse`
imports only `flight.libs` + `sim`.

**Tech Stack:** Python 3.14, uv workspace, numpy/scipy/structlog (flight), onnxruntime (lazy),
pytest, ruff, mypy --strict, import-linter.

---

## Cross-cutting design decisions (locked)

### Core-hosted services (spec §10 Approach A)
New services live in `flight.core` (top layer; may import apps + libs + hal.interfaces):
- `flight.core.command_router` — `CommandRouter`: routes `CommandMsg` → `RoutedCommandMsg`;
  ARM/EXECUTE two-step for hazardous commands; inhibit pre-check from published safety state;
  loud NACK + fault on unroutable; emits execution `CommandAckMsg`.
- `flight.core.storage` — `StorageService`: file-backed, checksummed, quota'd; implements the
  `StorageWriter` + `StorageReader` Protocols; bus-consumer run loop persisting telemetry/events;
  reboot-surviving fault ledger (append-only JSONL). **No filesystem I/O at `__init__`** (lazy dir
  creation) so `build_*` stays side-effect-free.
- `flight.core.downlink` — `DownlinkManager`: priority queue (fault > ack > HK telemetry >
  science product), AOS-gated + budget-gated, emits `DownlinkItemMsg` carrying inline bytes or a
  storage reference.

These are constructed by the composition roots (`flight.core.main`, `sim.sil`) into a new
`CoreServices` holder and run as additional `RunnableApp`s (flight) / stepped in `step_once` (SIL).
Each heartbeats and is added to `MONITORED_SUBSYSTEMS` where it runs a persistent loop.

### Storage Protocols location
`StorageWriter` / `StorageReader` Protocols go in **`flight.hal.interfaces.storage`** (apps depend
on injected Protocols; composition root injects the concrete `StorageService`). This keeps
`build_apps` driver-agnostic (Protocol-typed params only). `StorageService` (concrete, with I/O)
lives in `flight.core` and is constructed by the root.

### Command routing model
- `iss_iface` validates (existing) → publishes `CommandMsg` (ingress ACK unchanged).
- `CommandRouter` subscribes to `CommandMsg`, resolves `target` against a known routable set,
  applies hazardous ARM/EXECUTE, pre-checks inhibits, and republishes `RoutedCommandMsg` to the
  target app. Unknown/unroutable target → `CommandAckMsg(REJECTED, COMMAND_UNROUTABLE)` +
  `FaultEventMsg`.
- Target apps (`thermal`, `electrical`, `mechanical`, `fault`) consume **`RoutedCommandMsg`**
  (not raw `CommandMsg`) and emit an execution `CommandAckMsg(ACCEPTED/REJECTED)`. Apps enforce
  device interlocks at actuation (layered authority).
- Hazardous commands carry a `phase` STR param ("ARM"|"EXECUTE"). Router tracks armed state keyed
  by `(source, command_id)` with an arm timestamp; EXECUTE requires a prior ARM within
  `arm_window_s` and re-checks inhibits.

### Safety / inhibit state (fault-owned)
`FaultApp` becomes the SAFE-latch owner and publishes `SafetyStateMsg` each tick:
`{mode, active_faults, safe_latched, safe_reason}`. The router subscribes for inhibit pre-checks.
`EXIT_SAFE` targets `fault`; on EXECUTE the fault app publishes `ModeChangeMsg(IDLE)` only if no
SAFE-triggering fault was seen this tick (the "triggering fault cleared" gate) — the authoritative
inhibit enforcement at the actuator. The arbiter keeps its existing SAFE latch driven by
`ModeChangeMsg`.

### New enum / message / config additions (added incrementally per phase)
- `MessageType`: `ROUTED_COMMAND`, `SAFETY_STATE`, `LAUNCH_LOCK_STATE`, `MODEL_DEPLOY`.
- `FaultCode`: `COMMAND_UNROUTABLE`, `LAUNCH_LOCK_FAULT`.
- `CommandId`: `EXIT_SAFE`, `RELEASE_LAUNCH_LOCK`, `MANUAL_GIMBAL_SLEW`, `ACTIVATE_MODEL`,
  `STAGE_MODEL_CHUNK` (chunk uplink may bypass the dictionary — decided in Phase 5).
- `LaunchLockState` enum: `ENGAGED | RELEASED | UNKNOWN`.
- Messages: `RoutedCommandMsg`, `SafetyStateMsg`, `LaunchLockStateMsg`, `ModelDeployStateMsg`.
  Every message gains `schema_version: int = SCHEMA_VERSION` (defaulted; Phase 6).
- Config: `CommandRouterConfig` (arm_window_s, routable_targets), extend `CommsConfig`
  (downlink budget already present), `StorageConfig` (retention), `EnvironmentConfig` add inert
  `lock` axis handling note (Protocol+sim only; no real driver).

### Determinism / SIL
`step_once` gains: pump command router, storage service tick, downlink manager tick, mechanical
app, after the existing app steps and before/after the FDIR tick as ordering requires. The GSE
`TelemetryCapture` is extended to observe new evidence (routed/exec acks, downlink product count,
SAFE exit, lock state, model deploy state).

---

## Phase 1 — Config integrity (§7)

**Files:** Modify `packages/flight/src/flight/core/config_loader.py`;
Test `packages/flight/tests/test_config_loader.py` (extend or create).

- [ ] Write failing tests: out-of-range rejection (e.g. negative `fault.thermal_limit_c`,
  `ema_alpha` outside (0,1], `gimbal.az_min_deg >= az_max_deg`), cross-field (band indices vs
  mosaic layout: `inference.input_bands` ⊄ `sensor.mosaic_layout`; `mosaic_layout` not a
  permutation of Band), unknown-key rejection (typo'd section key + typo'd field key fail loudly).
- [ ] Implement full `_validate()`: per-section range checks, cross-field checks, unknown-key
  rejection (compare parsed keys against the known dataclass field sets). Keep returning `str|None`.
- [ ] Verify `test_config_defaults` (defaults-vs-TOML) still passes; `main()` honors override path
  (already wired — add a test that `load_config(default, profiles/sil.toml)` succeeds).
- [ ] Per-phase gates + commit.

## Phase 2 — Command router + ARM/EXECUTE + EXIT_SAFE (§6)

**Files:** Create `flight/core/command_router.py`, `flight/core/routing.py` (pure core);
Modify `flight/libs/messages/messages.py` (+RoutedCommandMsg, SafetyStateMsg),
`flight/libs/types/enums.py` (+ROUTED_COMMAND, SAFETY_STATE, COMMAND_UNROUTABLE, +CommandId
EXIT_SAFE), `flight/libs/commands/dictionary.py` (+EXIT_SAFE hazardous, phase param),
`flight/libs/config/config.py` (+CommandRouterConfig), `flight/core/config_loader.py`,
`config/default.toml`, `flight/fault/app.py` + `flight/fault/policy.py` (SAFE-latch ownership,
SafetyStateMsg, EXIT_SAFE handling), `flight/thermal/app.py` + `flight/electrical/app.py`
(consume RoutedCommandMsg + exec ack), `flight/core/composition.py` (CoreServices +
build_core_services), `sim/sil/stepping.py` (tick router), `sim/sil/runner.py`.

- [ ] Pure routing core tests + impl: `route(command, routable, armed_state, safety, now) ->
  RoutingDecision` (dispatch | reject | armed) — table of cases incl. unknown target, hazardous
  ARM then EXECUTE, EXECUTE without ARM, ARM expiry, inhibit blocks EXECUTE.
- [ ] `CommandRouter` shell: subscribe CommandMsg + SafetyStateMsg; tick() drains, applies route,
  publishes RoutedCommandMsg / CommandAckMsg(exec) / FaultEventMsg; heartbeats in run().
- [ ] FaultApp: own SAFE latch, publish SafetyStateMsg each tick, consume RoutedCommandMsg
  (EXIT_SAFE) gated on no active SAFE-fault → ModeChangeMsg(IDLE)+exec ack.
- [ ] thermal/electrical: switch CommandMsg→RoutedCommandMsg, emit exec CommandAckMsg.
- [ ] Composition: CoreServices holder, build_core_services, add router to MONITORED + scheduler +
  step_once; add "command_router"/"fault"(already)/services to monitored set.
- [ ] SIL test: command routed → executed → exec-acked; SAFE entered (thermal) then EXIT_SAFE
  exits to IDLE (gated). Per-phase gates + commit.

## Phase 3 — Data system: storage + downlink + fault ledger (§6)

**Files:** Create `flight/hal/interfaces/storage.py` (StorageWriter/StorageReader Protocols),
`flight/core/storage.py` (StorageService), `flight/core/downlink.py` (DownlinkManager);
Modify `flight/libs/config/config.py` (StorageConfig retention), `flight/payload/app.py` (inject
StorageWriter; store mask/thumbnail product), `flight/iss_iface/app.py` (inject StorageReader;
send DownlinkItemMsg only; resolve product refs), `flight/core/composition.py` (construct
StorageService, inject faces, build DownlinkManager), `sim/sil/*`, `gse/harness.py` (capture
downlink product count).

- [ ] StorageService unit tests + impl: store_product (sha256 sidecar + checksum verify on read),
  quota eviction (drop-oldest lowest-priority + drop counter; STORAGE_FULL on critical), fault
  ledger append + reboot-survive read, lazy dir creation.
- [ ] DownlinkManager unit tests + impl: priority ordering, AOS gate, byte budget gate, product
  ref vs inline.
- [ ] Wire payload→StorageWriter (store thumbnail of mask), iss_iface→StorageReader (resolve ref);
  refactor iss_iface downlink to send DownlinkItemMsg from DownlinkManager (acks flow via manager).
- [ ] SIL test + scenario: product stored → downlinked (count>0 on sil-link-real via emulator).
  Per-phase gates + commit.

## Phase 4 — Mechanical / LaunchLock (§5)

**Files:** Create `flight/hal/interfaces/launch_lock.py` (LaunchLock Protocol + LaunchLockState),
`flight/hal/drivers_sim/launch_lock.py` (SimLaunchLock), `flight/mechanical/app.py` (MechanicalApp);
Modify `flight/libs/types/enums.py` (+LAUNCH_LOCK_STATE, +LAUNCH_LOCK_FAULT, +CommandId
RELEASE_LAUNCH_LOCK), `flight/libs/messages/messages.py` (+LaunchLockStateMsg),
`flight/libs/commands/dictionary.py` (+RELEASE_LAUNCH_LOCK hazardous, target mechanical),
`flight/core/composition.py` (+mechanical app, MONITORED), `flight/core/select_drivers.py`
(+lock axis → SimLaunchLock; real deferred), `flight/payload/app.py` (gimbal interlock: refuse
motion while lock ENGAGED), `sim/sil/*`, `gse/*`.

- [ ] LaunchLock Protocol + SimLaunchLock unit tests + impl (ENGAGED→RELEASED).
- [ ] MechanicalApp: publish LaunchLockStateMsg each tick + telemetry + heartbeat; consume
  RoutedCommandMsg RELEASE_LAUNCH_LOCK; refuse release while gimbal moving (interlock) → exec ack.
- [ ] Bidirectional interlock: payload refuses gimbal motion while lock ENGAGED (subscribe
  LaunchLockStateMsg); add mechanical to Drivers + build_apps + select_drivers (inert real).
- [ ] SIL test + scenario: lock released via ARM/EXECUTE while idle; gimbal motion refused while
  ENGAGED. Per-phase gates + commit.

## Phase 5 — Model upload + model lifecycle (§6 + §4)

**Files:** Create `flight/iss_iface/upload.py` (chunk reassembly pure core),
`flight/core/model_deploy.py` (stage/ACTIVATE/rollback), `packages/tools/src/tools/accept.py`
(artifact-acceptance gate), CI fixture `packages/flight/tests/fixtures/tiny.onnx` (+generator);
Modify `flight/iss_iface/app.py` (consume UploadChunkMsg → reassemble → stage), `flight/libs/
messages/messages.py` (+ModelDeployStateMsg), `flight/libs/types/enums.py` (+MODEL_DEPLOY,
+CommandId ACTIVATE_MODEL), `flight/payload/model/detector.py` (OnnxDetector hash/contract verify
+ latency budget → INFERENCE_TIMEOUT), `flight/libs/commands/dictionary.py` (+ACTIVATE_MODEL
hazardous? — ground op, ARM/EXECUTE optional), `sim/sil/*`, `gse/*`.

- [ ] Chunk reassembly pure core tests + impl (ordered/duplicate/missing chunks, CRC verify).
- [ ] ModelDeploy core: stage → validate manifest+SHA-256+IO-contract → ACTIVATE swap → auto
  rollback on load/first-frame failure; ModelDeployStateMsg telemetry.
- [ ] tools/accept.py: manifest + SHA-256 + I/O contract + golden-scene IoU + latency gate
  (onnxruntime only; uses sim.scene + flight preprocess). Self-contained tests in tools/.
- [ ] OnnxDetector: load-time hash + I/O-contract verify (MODEL_CORRUPT); per-frame latency budget
  (INFERENCE_TIMEOUT, drop frame, continue). Tiny fixture .onnx behavioral test.
- [ ] SIL test + scenario: model uploaded → activated → rolled back. Per-phase gates + commit.

## Phase 6 — Platform robustness (§7)

**Files:** Modify `flight/libs/bus/bus.py` (bounded per-type queues + overflow policy),
`flight/libs/messages/messages.py` (schema_version on every message + SCHEMA_VERSION const),
`flight/core/scheduler.py` (thread supervision: restart-then-SAFE), `flight/core/main.py`
(SIGTERM ordered teardown, startup health-gate → SAFE), `flight/libs/config/config.py`
(bus bounds + supervision config), `config/default.toml`.

- [ ] Bus: per-message-type depth + overflow policy (drop-oldest+counter for telemetry; never-drop
  for command/fault → overflow is a FaultEventMsg). Unit tests for both policies + counter.
- [ ] schema_version: add `SCHEMA_VERSION` const + defaulted field on every message; one test
  asserts presence + value; update only constructions that must (defaulted → minimal churn).
- [ ] Scheduler supervision: detect dead thread, restart up to limit, then publish PROCESS_DIED →
  SAFE. Unit test with a crashing fake app.
- [ ] main(): SIGTERM handler → ordered teardown (quiesce payload → drain downlink → flush storage
  → join). Startup health-gate: wait for all MONITORED heartbeats within window else SAFE. Unit
  test the health-gate decision as a pure helper.
- [ ] Per-phase gates + commit.

## Phase 7 — VCRM rows + scenarios + final whole-tree gates

**Files:** Add scenarios under `scenarios/` (command_route_exec, safe_exit, product_downlink,
launch_lock_release, model_activate_rollback); Modify `docs/requirements/vcrm.toml` + `vcrm.md`
(new running-venue rows with module + evidence), `gse/tests/test_scenarios.py`.

- [ ] Author scenarios exercising each new capability; add them to `gse/tests/test_scenarios.py`.
- [ ] Add VCRM rows (new REQ-IDs) cited by module docstrings (`Satisfies:`) + scenario/test
  evidence; `scripts/check_vcrm.py` exit 0.
- [ ] Run ALL whole-tree gates: ruff check / format --check / mypy / lint-imports / check_vcrm /
  pytest -m "not e2e". Final commit. Update memory `project_fsw_parity_effort.md`.

---

## Self-review (spec coverage)
- §6 command router → Phase 2. §6 data system → Phase 3. §6 model upload → Phase 5.
- §4 model lifecycle (tools gate + OnnxDetector) → Phase 5.
- §5 mechanical/LaunchLock → Phase 4.
- §7 platform robustness → Phase 6. §7 config integrity → Phase 1.
- SIL/GSE exercise + VCRM → woven into each phase + consolidated in Phase 7.
- Out of scope (per goal): PIL/HIL run, real LaunchLock/RealScalarSensor drivers, systemd, full
  §8 requirements/hazard docs (only VCRM rows the new scenarios need).
