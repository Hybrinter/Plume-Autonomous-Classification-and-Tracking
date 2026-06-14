# PACT Software -- Agent Context

This file contains non-obvious, cross-cutting patterns for the PACT codebase.
These apply project-wide and cannot be derived by reading individual files.
For system architecture and design rationale, see `docs/architecture.md` and the design spec
`docs/superpowers/specs/2026-05-30-pact-iss-payload-fsw-structure-design.md`.

---

## Where the Code Lives

The flight software is the `uv` workspace under `packages/`:

- `packages/flight/` (`pact-flight`) -- the flight software (lean deps: numpy, scipy, structlog).
- `packages/sim/` (`pact-sim`) -- SIL harness, scene generation, digital twin (depends on flight).
- `packages/tools/` (`pact-tools`) -- training/eval/export; heavy deps (torch etc.) live here only.

The legacy `src/pact/` tree (the pre-restructure multiprocessing/`ops/main.py` codebase) has been
**removed**; `packages/` is the entire codebase. PACT is an ISS-attached payload and the codebase
is Python-only (the earlier Rust-migration plan was dropped).

Run gates over the whole tree: `uv run ruff check packages scripts`, `uv run ruff format --check
packages scripts`, `uv run mypy packages scripts`, `uv run lint-imports`, `uv run python
scripts/check_vcrm.py`, and `uv run pytest -m "not e2e"`. The repo root is a virtual `uv` workspace
root (`[tool.uv] package = false`); it builds no package and carries no runtime deps of its own.

---

## Subsystem-App + Bus Contract

Each subsystem under `packages/flight/src/flight/` is an isolated **app**: a thin imperative shell
around a pure core, talking to other apps **only** over the typed `MessageBus`
(`flight.libs.bus`). No app imports or references another app. Peer apps
(`payload`/`fault`/`iss_iface`/`thermal`/`electrical`) must never cross-import -- this is enforced
by the `flight-layers` import-linter contract (layer order: `core` > apps > `hal.interfaces` >
`libs`).

**Invariant:** if a new inter-subsystem channel is needed, it is a message type in
`flight.libs.messages` published/subscribed on the bus -- never a direct call or a queue an app
constructs itself.

---

## Composition-Root Ownership

`flight.core` is the only composition root for flight (and `sim.sil` for SIL). It alone:
constructs the `MessageBus`, the `Clock`, and the concrete HAL drivers; calls
`flight.core.composition.build_apps(config, bus, clock, drivers, monitored)` to wire every app;
and runs them via `flight.core.scheduler.Scheduler` (one daemon thread per app's
`run(stop_event)`).

**Invariants:**
- Only composition roots construct the bus and the drivers. Apps receive injected bus
  subscriptions + driver Protocols as constructor arguments. Pure cores touch neither.
- `build_apps` and `flight.core.scheduler` must stay **driver-agnostic** -- they import only HAL
  Protocols and apps, never `flight.hal.drivers_real`/`drivers_sim`. Only `flight.core.main`
  imports concrete real drivers. This is enforced by the `drivers-from-composition-roots-only`
  import-linter contract; keeping `build_apps` driver-free is what lets the SIL reuse it verbatim.

---

## Preprocessing Co-Location Invariant

Preprocessing runs as plain function calls inside `PayloadApp.process_frame()`
(`flight/payload/app.py`), not as a separate process/thread and not across the bus.

**Why:** preprocessing outputs a `(C, H, W)` float32 numpy array. Putting it on the bus (or any
queue) requires pickling -- a per-frame serialization cost. As an in-function value it has zero
overhead.

**Invariant:** never publish `ProcessedFrameMsg` to the bus; never add a process/thread boundary
between preprocessing and inference. New preprocessing logic goes in `flight/payload/preprocess/`
as a pure function called from `process_frame()`. (Large artifacts -- tensors, masks -- never go
on the bus; only compact records do.)

---

## Pure-Core Contract (controller, arbiter, tracking, watchdog, policy)

The decision cores are **pure functions**: no I/O, no bus access, no clock reads, no logging. They
map inputs (including `now` and current state) to outputs (new state + messages) deterministically.
This holds for `PayloadController.step`, `GimbalArbiter.step`, the tracking estimators
(`ema_update`, Kalman `predict`/`update`, `match_blobs`), the LQR law, and the FDIR
`check_heartbeats` / `decide_mode_change`.

**Why:** pure cores are trivially unit-testable, replayable from logs, and free of concurrency
concerns. All mutable state is threaded in and out (passed in, returned out); the app shell owns
the bus, the clock, and the state variable.

**Invariant:** never add I/O, bus access, side effects, or a clock source inside a pure core. Time
is passed in as a `now: float` argument (monotonic seconds). Any new core logic must be expressible
as a pure state transformation.

---

## Result[T, E] Usage Contract

Library code returns `Result[T, E]` (`Ok` | `Err`, from `flight.libs.types`) -- it never raises.
Process entry points may raise for unrecoverable startup failures only (e.g. bad config in
`main`).

**The distinction:** if a caller can meaningfully handle the failure (retry, degrade, emit a
fault), it is a `Result`. If the system cannot continue without human intervention, it is a startup
exception.

**Pattern:**

```python
result = some_library_function(...)
if isinstance(result, Err):
    bus.publish(FaultEventMsg(..., fault_code=result.error))
    return
value = result.value  # safe: narrowed to Ok
```

Never read `.value` without an `isinstance(result, Ok)` (or `Err`) check first.

---

## HAL Protocol + Lazy-SDK Contract

Every device class is a `@runtime_checkable` `Protocol` in `flight.hal.interfaces`, returning
`Result[..., FaultCode]`. Real drivers (`drivers_real`) and sim drivers (`drivers_sim`) implement
it structurally; the composition root injects the choice and **apps never know which they got**.

Hardware/ML SDK imports are **lazy** -- inside `__init__`/`detect`, not at module top. Importing a
driver module never requires its SDK; only constructing the real driver does (`RealSensor` ->
PySpin, `OnnxDetector` -> onnxruntime). This keeps CI and the lean flight image SDK-free. The real
and sim driver sets must not import each other (`drivers-independent` contracts).

---

## Config Distribution

`flight.core.config_loader.load_config()` loads `config/default.toml` (merged with an override if
present) once at startup, producing a frozen `PactConfig`. Each subsystem's `from_config` receives
the typed config it needs -- no subsystem reads TOML directly.

**Invariant:** default field values in `flight/libs/config/config.py` must match
`config/default.toml` exactly (a divergence is a silent test-reproducibility bug; the
`test_config_defaults` test guards this).

---

## Heartbeat / Watchdog Contract

Every app that runs a persistent loop emits `HeartbeatMsg` periodically (in its `run()` loop, every
`watchdog_interval_s`, default 5 s). The FDIR watchdog (`flight.fault`) monitors the subsystems in
`flight.core.composition.MONITORED_SUBSYSTEMS` and emits `WATCHDOG_EXPIRE` after
`watchdog_max_miss_count` (default 3) consecutive misses; `flight.fault.policy` routes that (and
the other SAFE-triggering faults) to `ModeChangeMsg(SAFE)`.

**Implementation pattern:** loops use `stop_event.wait(timeout=interval)`, not `time.sleep`, so
shutdown is immediate. The deterministic SIL harness stands in for these per-app heartbeats by
publishing them itself each step.

---

## Strong Typing + mypy_path

mypy runs `--strict`. The root `pyproject.toml` sets
`mypy_path = ["packages/flight/src", "packages/sim/src", "packages/tools/src", "packages/gse/src"]`
so cross-package `flight.*`/`sim.*`/`tools.*`/`gse.*` imports resolve to the workspace **source**
trees. **Do not remove it** -- without it those imports fall back to `Any` (the editable installs have no `py.typed`),
silently disabling strict checking across modules. Polymorphism is expressed with statically-typed
`Protocol` interfaces (the relaxed form of the "no dynamic dispatch" rule); avoid callable dispatch
tables and duck typing. See `.claude/rules/strong_typing.md`.

---

## Subsystem Context Files

Detailed non-obvious context per subsystem (read on demand when working in one):

- `packages/flight/src/flight/libs/CONTEXT.md`
- `packages/flight/src/flight/hal/CONTEXT.md`
- `packages/flight/src/flight/core/CONTEXT.md`
- `packages/flight/src/flight/payload/CONTEXT.md`
- `packages/flight/src/flight/fault/CONTEXT.md`
- `packages/flight/src/flight/iss_iface/CONTEXT.md`
- `packages/flight/src/flight/thermal/CONTEXT.md` (also covers `electrical`)
- `packages/sim/src/sim/CONTEXT.md`
