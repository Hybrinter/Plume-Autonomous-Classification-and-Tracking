# PACT ISS Payload -- Flight Software Structure (Design Spec)

- Date: 2026-05-30
- Status: Approved for planning
- Scope: Repository structure, build tooling, subsystem architecture, interfaces,
  conventions, and the implementation sequence. This spec defines the *skeleton and
  contracts*, not the implementation of any individual subsystem.

---

## 1. Project framing

PACT is an independent, **ISS-attached payload** for autonomous plume detection,
segmentation, and tracking. It is not a free-flying satellite and is not related to any
employer's program. The flight software runs on a **single Linux flight computer**.

The station provides the true bus services -- primary power, thermal sink, attitude, and
the downlink path to the ground. PACT therefore does not implement a free-flyer bus; its
"non-payload" subsystems are scoped to the payload's *own* housekeeping plus the interface
to the station.

### Goals

1. **Real-FSW rigor** -- a canonical layout, clean interfaces, strong typing, and
   testability that mirror professional flight software.
2. **Experiment velocity** -- a simulation environment that lets the real flight code run
   against a simulated world, so algorithm experiments are fast and repeatable.

### Non-goals / explicitly dropped

These were considered and deliberately removed; recorded here so future readers do not
re-introduce them by assumption:

- **Rust migration.** The codebase will remain Python. The payload's heavy math is already
  thin glue over C/C++/CUDA libraries (numpy, ONNX Runtime, OpenCV); the flight-critical
  control/safety glue does not benefit enough from a systems language to justify the cost,
  given the relaxed reliability posture below. All constraints that existed *only* to make
  Rust translation mechanical are removed (see Section 12).
- **Bazel.** With a single language there is no polyglot build to justify Bazel. Standard
  Python tooling is used instead, with `import-linter` replacing Bazel `visibility` for
  layering enforcement (Section 11).

### Reliability and safety posture

Because the payload is continuously commandable from the ground through the station, is
recoverable, and sits inside the station's safety nets, the **software** reliability target
is lower than a free-flyer: design for **fail-safe, ground-recoverable, graceful
degradation** rather than heavy autonomous fault tolerance or redundancy.

One caveat is not relaxed: anything attached to a crewed vehicle has NASA payload **safety**
requirements for hazardous functions (stored mechanical energy in the gimbal, batteries, RF
emission, thermal limits affecting the host). That is primarily a hardware/operations safety
review; the software's obligations are narrow and explicit -- **honor safety inhibits and
never command a hazardous function without authorization.**

---

## 2. Architecture overview -- the subsystem-app model

Each subsystem is an **app**: one OS process, one Python package, isolated behind a message
bus. The model is borrowed from the canonical NASA cFS app pattern (apps + software bus)
without adopting any framework code.

Three rules define an app:

1. **Isolation.** An app imports only `flight.libs.*` and `flight.hal.interfaces`. It never
   imports another app. All inter-app coupling is *message types* defined in `flight.libs.messages`.
2. **Pure core + thin shell.** Each app splits into a pure-logic core (deterministic, no
   I/O -- the existing `GimbalArbiter.step` is the template) and a thin process shell that
   owns the loop, the bus handles, the clock, and the injected driver. The pure core is the
   unit-test target.
3. **No self-owned channels; heartbeat.** An app creates no queues; `flight.core` creates
   every channel and injects publish/subscribe handles. Every app emits the standard
   heartbeat (`threading.Event.wait(timeout=...)`, not `time.sleep`, for clean shutdown).

`flight.core` is the **Compute / C&DH host** -- not a peer app. It is the composition root:
it loads config, builds the configured driver set, spawns each app with injected
dependencies, and hosts the scheduler, the system clock, the bus router, persistent storage,
the telemetry-downlink aggregator, and FDIR coordination. Today's `ops/` and `storage/`
grow into it.

### Layering

```
L0  flight.libs.types, flight.libs.messages          (pure data)
L1  flight.libs.bus, flight.libs.time, flight.libs.telemetry, flight.hal.interfaces
L2  flight.{payload, thermal, electrical, mechanical, iss_iface, fault}   (peer apps, no cross-imports)
L3  flight.core                                       (composition root; hosts L2; drivers linked here)
```

The single invariant that makes apps independently testable: **L2 apps are siblings that
only ever communicate through message types in L0.**

---

## 3. Repository layout

```
repo-root/                          # uv workspace
  pyproject.toml  uv.lock           # workspace root; ruff + mypy + pytest config
  importlinter.ini                  # layering contracts (replaces Bazel visibility)
  packages/
    flight/                         # lean flight package (deps: numpy, onnxruntime, structlog, toml)
      core/                         # Compute/C&DH host: composition root, scheduler, clock,
                                    #   bus router, storage, telemetry aggregator, FDIR coordination
      payload/                      # acquire -> preprocess -> infer -> tracking -> gimbal arbiter
      thermal/                      # payload's own heaters + temperature sensors
      electrical/                   # payload power conditioning off the ISS feed
      mechanical/                   # aperture covers / deployables / latches  (NOT the gimbal)
      iss_iface/                    # command + telemetry + data exchange with the station/ground
      fault/                        # pragmatic FDIR (fail-safe oriented)
      libs/                         # types, messages (frozen dataclasses), bus, time, telemetry
      hal/                          # Protocol-based device interfaces + real/sim drivers
    sim/                            # separate package: sil/ scene/ twin/ (plant model)
    tools/                          # separate package (deps: torch, matplotlib, ...): experiments, training, replay
  config/                           # default.toml + flight.toml; per-subsystem typed config tables
  docs/                             # architecture.md, adr/, requirements/
  tests/                            # integration/ and e2e/ (unit tests co-located per package)
```

Splitting `flight` / `sim` / `tools` into separate workspace packages keeps the **flight
environment lean** -- no torch or matplotlib on the flight computer; training and analysis
dependencies live only in `tools`.

---

## 4. Dependency spine and layering enforcement

The dependency direction is the spine of the system, enforced by `import-linter` contracts
(and reviewed in CI):

- `flight.*` may import only `flight.libs.*` and `flight.hal.interfaces`. Never another app,
  never a concrete driver, never `sim` or `tools`.
- `flight.hal.drivers_real` and `flight.hal.drivers_sim` are reachable **only** from
  composition roots: `flight.core` (flight) and `sim.sil` (simulation). This is the
  dependency-inversion seam -- apps know the interface, the composition root injects the
  implementation.
- `sim.*` may import `flight.*` and `flight.hal.drivers_sim` (it drives the real apps against
  fake hardware). `flight` never imports `sim`.
- `tools.*` may import anything (it is the workshop).
- `flight.libs.*` imports only other `flight.libs.*`, in layer order.

---

## 5. The message bus and contract

Generalize today's `*Msg` + `multiprocessing.Queue` pattern into a **typed pub/sub bus**
whose API is transport-agnostic:

- `bus.publish(msg)` / `bus.subscribe(MsgType) -> handle`. Routing is **by message type**;
  each message type is its own topic. `flight.core` owns the router (a
  `type -> subscriber-queues` registry).
- **Transport is `multiprocessing.Queue` (pickle), permanently.** With a single language and
  no Rust seam to plan for, there is no serialization contract to maintain and no future
  flip. (If a single large-array channel ever needs it, shared memory can be introduced
  behind the same `publish`/`subscribe` API without touching app code.)
- Queue-ownership invariant unchanged: **only `flight.core` constructs queues and the
  router**; apps receive injected handles. Pure cores never touch the bus.

Three standard message envelopes give the canonical "everything is commandable, everything
is telemetered" FSW property:

- `CommandMsg{target, command_id, params, source, seq}` -- ground/station -> `iss_iface`
  -> `core` -> target app.
- `TelemetryMsg{subsystem, fields, t}` -- app -> `core` -> `iss_iface` downlink + `storage`.
- Events/faults: today's `FaultEventMsg` and `HeartbeatMsg` become specializations of an
  event envelope.

**Large artifacts (segmentation masks, raw frames) never go on the bus** -- they go to
`storage` directly. The bus carries compact detection/track records only. This preserves the
same serialization-cost discipline as the preprocessing co-location invariant.

Messages are frozen dataclasses in `flight.libs.messages`. There is no longer a "wire-safe
subset" rule; messages are constrained only by ordinary good taste and the
strong-typing conventions in Section 12.

---

## 6. The hardware abstraction layer (HAL)

Each device class is a statically-typed **`Protocol`** in `flight.hal.interfaces` (e.g.
`ImagingSensor`, `GimbalActuator`, `Heater`, `PowerChannel`, `StationLink`, `Mechanism`).
Real and simulated drivers implement the protocol; the composition root selects the
implementation from config. (The former enum-of-drivers pattern, which existed to avoid
dynamic dispatch for Rust, is replaced by clean `Protocol`-based dispatch -- see Section 12.)

```
flight/hal/interfaces:   class ImagingSensor(Protocol): def read_frame(self) -> Result[Frame, Fault]: ...
flight/hal/drivers_real: RealSensor, RealGimbal, RealHeater, RealStationLink, ...
flight/hal/drivers_sim:  SimSensor, SimGimbal, ...   (fed by sim.scene + sim.twin)
```

The composition root picks the implementation:
- `flight.core` (flight build) constructs the real drivers selected by config.
- `sim.sil` (simulation) constructs the sim drivers. **Apps never know which they got.**

---

## 7. Inside the payload app

The payload's stages are **internal stages of one process**, satisfying the preprocessing
co-location invariant (a `(C, H, W)` float32 tensor is never pickled across a process
boundary):

```
flight/payload/
  process.py        # app shell: loop, bus handles, injected sensor + gimbal drivers
  acquire/          # frame acquisition via the sensor driver (HAL)
  preprocess/       # pure fn(s) -> (C,H,W) float32      [co-located with infer, no queue]
  model/            # inference: frozen model artifact + runtime session; swappable backend
  tracking/         # pure tracking controller (mode FSM + estimator + control law)
  gimbal/           # pure gimbal arbiter (today's GimbalArbiter), resolves command sources
```

Loop, all in one process:
`acquire -> preprocess -> infer -> tracking.step(...) -> arbiter.step(...) -> command the
gimbal driver + publish compact detection/track telemetry`.

### The onboard model

- The model is a **frozen, versioned artifact** consumed by the flight code, never trained
  in flight. Training stays entirely in `tools/` (free-form torch). The artifact's
  version/hash is recorded in telemetry for configuration control.
- Recommended packaging: **export to ONNX, run via `onnxruntime`** -- a frozen artifact,
  lean flight deps (no torch on the flight computer), often faster inference, deterministic.
  Running a frozen torch module directly is an acceptable alternative; this is no longer a
  load-bearing decision now that Rust is off the table.
- The **detector backend is swappable** behind a small interface
  (`OnnxDetector` for the real model, `ScriptedDetector` for deterministic detections in
  SIL), selected by config. This makes SIL runs deterministic and fast, and decouples
  algorithm experiments from the live model.
- The previous "one intentional exception" (a frozen dataclass wrapping a mutable
  `torch.nn.Module`) is **retired**: the model is a read-only artifact behind a runtime
  session, not mutable module state held in a struct.

### The tracking controller

A hybrid that stays entirely pure-function and unit-testable:

- A **discrete mode FSM** on top (e.g. `SEARCH -> ACQUIRE -> TRACK -> REACQUIRE -> SAFE`):
  an enum plus a pure transition function.
- Inside `TRACK`, a **target-state estimator** (a track filter over detections) and a
  **control law** (turns the estimate plus current gimbal state into a pointing/rate request).
- Signature mirrors the existing pure-arbiter contract:
  `TrackingController.step(state, observations, now) -> Result[(TrackingState, [GimbalRequest | ModeEvent]), Fault]`,
  where `observations` bundle detections + gimbal telemetry + system state.

The tracking controller and the existing `GimbalArbiter` are two tiers: the **tracking
controller is a command source** (it requests pointing), the **arbiter is the resolver**
(it arbitrates tracking requests against ground overrides, safe-mode, and limits into the
final gimbal command). Both stay pure.

**Gimbal split:** the gimbal *actuator* is a HAL device; its *pointing control* (tracking +
arbiter) lives in `payload`. The `mechanical` subsystem handles covers, deployables, and
latches -- not the gimbal.

---

## 8. Simulation and closed-loop SIL

`sim.sil` is the simulation composition root: it builds the real flight apps and core wiring
but injects `drivers_sim`, fed by `sim.scene` (plume/scene generation) and `sim.twin` (the
plant model). The flight payload app runs **unchanged** against a simulated world.

The plant model lives **only in simulation** (`sim.twin`) -- the flight controller carries no
internal dynamics model. This closes the tracking loop:

```
tracking controller -> gimbal request -> arbiter -> SimGimbal -> sim.twin advances gimbal+plume dynamics
        ^                                                                    |
        |                                                                    v
   detections <- model <- preprocess <- SimSensor <- sim.scene renders the new plant state
```

Experiments (in `tools/`) are then "run `sim.sil` with config/scene/plant variations, collect
telemetry, analyze" -- with the flight code never aware it is in a sim.

A replay driver (recorded data fed through the same HAL seam) is a future `drivers_sim`
variant.

---

## 9. Configuration and tables

- `flight.core.config_loader` loads `config/default.toml` (merged with `config/flight.toml`
  if present) once into a frozen `FlightConfig` with a **typed sub-config per subsystem**;
  each app receives only its slice. This generalizes the existing config-distribution
  invariant.
- Driver selection (real/sim per device) and the detector backend (Onnx/Scripted) are config
  fields read by the composition root.
- Runtime-tunable parameters ("tables") live in TOML; the canonical defaults mirror the
  dataclass defaults.
- **New CI check** (the one CLAUDE.md notes is missing): a test asserting that every config
  dataclass default equals `config/default.toml`. This kills the silent-divergence bug class.

---

## 10. Reliability, fault management, and safety

- `flight.fault` owns FDIR logic; `flight.core` hosts the coordinator. The watchdog/heartbeat
  contract is retained (heartbeat per app; misses trigger a fault).
- The posture is **fail-safe and ground-recoverable**: on an unrecoverable subsystem fault,
  bring the payload to a defined SAFE state (stow the gimbal, close covers, stop emitting),
  emit telemetry, and wait for ground command -- rather than attempting heavy autonomous
  recovery.
- **Safety inhibits** for hazardous functions are explicit, checked, and never bypassed by
  autonomy. This is the one place rigor is not relaxed.

---

## 11. Tooling and build

- **uv workspace** with three packages: `flight` (lean: numpy, onnxruntime, structlog,
  toml), `sim`, and `tools` (torch, matplotlib, analysis). Per-package dependency sets keep
  the flight environment lean. Lockfile (`uv.lock`) for reproducible environments.
- **Quality gates:** `ruff` (lint + format), `mypy` (or pyright) for types, `pytest` for
  tests, `import-linter` for the layering contracts in Section 4.
- **CI** runs: build/install, lint, type-check, layering check, the config-default check,
  and the test suite.

---

## 12. Conventions and typing rules

### Carried forward (good Python regardless of language)

- `@dataclass(frozen=True, slots=True)` for data-carrying structs; `dataclasses.replace()`
  for modified copies.
- `Result[T, E]` for library code (it never raises); process entry points may raise only for
  unrecoverable startup failures.
- Enums whose string value mirrors the member name (`IDLE = "IDLE"`).
- Full type annotations everywhere; no `**kwargs` (no `*args` except test helpers).
- `structlog` everywhere with `subsystem` and `event` structured fields; JSON renderer for
  flight, console for development.
- Numpy dtype/shape comments at declaration sites; line length 100; grouped imports
  (stdlib -> third-party -> internal); module docstrings cite requirement IDs; docstring
  rules per `.claude/rules/docstrings.md`.
- Pure-core + thin-shell per app; queue/bus creation only in `flight.core`; heartbeat per app;
  preprocessing co-location in the payload process.

### Changed by this design

- The **Rust-Migration Contract** section of CLAUDE.md is **removed**.
- The **`no dynamic dispatch` / `no duck typing`** rule in `.claude/rules/strong_typing.md`
  is **relaxed**: statically-typed `Protocol`-based interfaces are allowed and preferred for
  the HAL and strategy patterns. Untyped duck typing remains discouraged.
- The wire-safe message subset and serialized-transport seam are **not adopted**.
- The `InferenceEngine` frozen-torch exception is **retired** (Section 7).
- Layering is enforced by `import-linter`, not prose.

### Documentation to produce

- Rewrite `docs/architecture.md` to the new topology and rationale.
- Rewrite `CLAUDE.md` to drop the Rust contract and reflect the new structure and tooling.
- ADRs for: subsystem-app model; the message bus and envelopes; the `Protocol`-based HAL;
  closed-loop SIL with the plant model in sim; the ISS-payload subsystem reinterpretation;
  Python-only (drop Rust and Bazel); the reliability posture.
- Per-subsystem `CONTEXT.md` under each `flight/<subsystem>/`.
- `docs/requirements/` with one document per subsystem; module docstrings cite `REQ-*` IDs.

---

## 13. Testing strategy

- **Unit:** co-located tests per package, exercising the pure cores (FSM transitions,
  arbiter, estimator) -- fast, no I/O.
- **Integration (`tests/integration/`):** an app plus its sim driver.
- **End-to-end (`tests/e2e/`):** the whole flight image via `sim.sil` against `sim.scene`.
  Today's `test_full_pipeline_smoke.py` becomes this closed-loop SIL test.
- **Experiments (`tools/`):** runners that spin up SIL with varied config/scene/plant,
  collect telemetry, and analyze. Not flight code; unconstrained dependencies.

---

## 14. Implementation sequence (for the migration workflow)

This spec defines the structure; the implementation plan (next step) details the work. The
intended order:

1. uv workspace + packaging + `ruff`/`mypy`/`pytest` + `import-linter` layering contracts +
   CI + the config-default check.
2. `flight/libs/` -- types, messages (frozen dataclasses), bus, time, telemetry.
3. `flight/hal/` -- `Protocol` interfaces + real/sim drivers; route today's sensor/gimbal
   access through the HAL.
4. `flight/core` -- from today's `ops/` + `storage/`: composition root, scheduler, clock,
   bus router, storage, telemetry aggregator, FDIR home.
5. `flight/payload` -- re-home model/preprocess/imaging/controller; detector backend
   (ONNX/scripted); tracking (FSM + estimator + control law); preserve co-location.
6. `flight/iss_iface` and `flight/fault` -- station command/telemetry interface and pragmatic
   FDIR; standardize the command/telemetry/event envelopes.
7. `flight/thermal`, `flight/electrical`, `flight/mechanical` -- minimal-but-real apps
   (heartbeat + telemetry + commandable no-op) so the topology is provable end-to-end.
8. `sim/sil` + `sim/scene` + `sim/twin` (plant model); convert the e2e smoke test to
   closed-loop SIL.
9. Docs -- ADRs, per-subsystem `CONTEXT.md`, requirements, rewrite `architecture.md` and
   `CLAUDE.md`.

---

## 15. Open questions / future work

- Exact ISS data-interface protocol for `iss_iface` (depends on the chosen external facility
  and avionics interface). Designed behind the `StationLink` HAL protocol so it can be
  specified later.
- Whether `thermal`/`electrical`/`mechanical` grow beyond minimal apps depends on the actual
  payload hardware; the architecture supports either.
- A replay `drivers_sim` variant for flight-data regression once recorded data exists.

---

## 16. Decision log

Decisions captured during brainstorming (2026-05-30):

- Scope: full payload flight software, subsystem-app architecture.
- Compute: single Linux flight computer.
- Backbone: generalize the current typed-Msg bus (in-process queues).
- Simulation: SIL + HAL first, full digital twin (plant model) later.
- Structure: subsystem-app monorepo (Approach A).
- Onboard model: frozen artifact via a runtime (ONNX recommended), swappable detector backend.
- Tracking: discrete mode FSM + estimator + control law, all pure.
- Plant model: simulation only (`sim.twin`).
- Message contract: frozen dataclasses (no IDL).
- Language: **Python-only** (Rust dropped).
- Build: **uv workspace + import-linter** (Bazel dropped).
- Typing rule: relax `no dynamic dispatch` to allow `Protocol`-based interfaces.
- Deployment: ISS-attached payload; fail-safe / ground-recoverable reliability posture;
  honor NASA payload safety inhibits.
- Gimbal: pointing control in `payload`; `mechanical` = covers/deployables/latches.
- `thermal`/`electrical`/`mechanical`: minimal-but-real apps, not placeholders.
