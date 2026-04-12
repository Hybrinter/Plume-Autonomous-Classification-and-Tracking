# PACT Software -- Agent Context

This file contains non-obvious, cross-cutting patterns for the PACT codebase.
These apply project-wide and cannot be derived by reading individual files.
For system architecture and design rationale, see `docs/architecture.md`.

---

## Rust-Migration Contract

Every Python design choice optimizes for mechanical translation to Rust. The codebase is
written Rust-idiomatically (frozen dataclasses, enum discriminants, `Result[T,E]` error
handling, no dynamic dispatch) so that the translation is a structural mapping, not a
rewrite. The typing rules themselves are in `.claude/rules/strong_typing.md`.

The one intentional exception: `InferenceEngine` is `@dataclass(frozen=True)` but holds a
mutable `torch.nn.Module`. The frozen constraint prevents field *reassignment* -- it cannot
prevent in-place weight mutation. Weights must not change after construction; this is
enforced by convention only.

---

## Preprocessing Co-Location Invariant

Preprocessing runs as a plain function call inside `_run_inference_process()` in
`ops/main.py`, not as a separate process or thread. This is the most non-obvious
structural decision in the codebase.

**Why:** Preprocessing outputs a `(C, H, W)` float32 numpy array. Passing this through a
`multiprocessing.Queue` requires pickling -- a significant serialization cost on every
frame. As a function call, it has zero overhead.

**Invariant:** Never move preprocessing to a separate process or thread. Never add a
`ProcessedFrameMsg` queue between preprocessing and inference. If you're adding
preprocessing logic, it goes in `src/pact/preprocessing/` as a pure function, called from
`_run_inference_process()`.

---

## Pure-Function Arbiter Contract

`GimbalArbiter.step(state, result, now)` is a pure function -- no side effects, no I/O,
no queue access, no logging. It maps inputs to outputs deterministically.

**Why:** Pure functions are trivially unit-testable, replayable from logs, and translatable
to Rust without concurrency concerns. All mutable state lives in `ArbiterState` (passed in,
returned out). The caller (`controller/process.py`) owns the queues, the clock, and the
state.

**Invariant:** Never add I/O, queue access, or side effects to `GimbalArbiter`. Never move
the clock source inside the arbiter. Any new arbiter logic must be expressible as a pure
state transformation.

---

## Result[T, E] Usage Contract

Library code returns `Result[T, E]` -- it never raises. Process entry points may raise for
unrecoverable startup failures only.

**The distinction:** if a caller can meaningfully handle the failure (retry, degrade, emit
a fault), it's a `Result`. If the system cannot continue without human intervention (bad
config, missing model file), it's a startup exception.

**Pattern:**

```python
result = some_library_function(...)
match result:
    case Ok(value):
        # use value
    case Err(fault_code):
        fault_queue.put(FaultEventMsg(..., fault_code=fault_code))
```

Do not call `.value` without checking `isinstance(result, Ok)` first.

---

## Queue Ownership

All inter-process queues are created in `ops/main.py:main()` and passed as constructor
arguments to each subsystem's process entry point. No subsystem creates its own queues.

**Why:** Keeping queue creation in one place makes the full process topology visible without
reading 10 files. It also prevents a subsystem from silently creating a local queue that
never reaches its intended consumer.

**Invariant:** Never create a `multiprocessing.Queue` or `queue.Queue` inside a subsystem
module. If a new inter-subsystem channel is needed, add it to `ops/main.py`.

---

## Config Distribution

`ops/config_loader.py` loads `config/default.toml` (merged with `config/flight.toml` if
present) once at startup, producing a frozen `PactConfig` instance. Each subsystem receives
its typed config dataclass as an argument -- no subsystem reads TOML directly.

**Invariant:** Default field values in `src/pact/types/config.py` must match
`config/default.toml` exactly. There is no CI check for this yet -- divergence is a silent
bug that affects test reproducibility.

---

## Heartbeat Contract

Every subsystem that runs as a persistent process or thread sends `HeartbeatMsg` periodically
to `heartbeat_queue`. The fault watchdog expects a heartbeat every `watchdog_interval_s`
(default 5 s); three consecutive misses triggers `FaultCode.PROCESS_DIED`.

**Implementation pattern:** use `threading.Event.wait(timeout=watchdog_interval_s)` not
`time.sleep(watchdog_interval_s)`. The `wait()` call exits immediately when `stop_event` is
set, enabling clean shutdown without waiting out the full interval.

---

## Subsystem Context Files

Detailed non-obvious context for each subsystem. Not auto-loaded -- read on demand when
working in a subsystem.

- `src/pact/types/CONTEXT.md`
- `src/pact/model/CONTEXT.md`
- `src/pact/preprocessing/CONTEXT.md`
- `src/pact/controller/CONTEXT.md`
- `src/pact/imaging/CONTEXT.md`
- `src/pact/comms/CONTEXT.md`
- `src/pact/storage/CONTEXT.md`
- `src/pact/telemetry/CONTEXT.md`
- `src/pact/fault/CONTEXT.md`
- `src/pact/ops/CONTEXT.md`
