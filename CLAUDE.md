# PACT Software — Top-Level Project Context

## §1 Project Context

PACT is an ISS external payload that autonomously detects industrial smoke stack plumes from
~420 km orbital altitude, drives an active gimbal to track them, and downlinks imagery + metadata
for ground-based ML retraining. The onboard compute is a **Nvidia Jetson Xavier**.

**Phase I (only phase in scope for this software package):**
- Run a pretrained U-Net/ResNet-34 segmentation model on multispectral VNIR imagery
- Drive a gimbal using a 4-state arbiter safety system
- Store imagery + metadata with checksums and manifests
- Downlink over CCSDS packet protocol via TDRSS (weekdays only, 5 Mbps down / 2 Mbps up)
- Support safe model uplink and staged deployment with rollback

**Key hard limits that affect software design:**
- Inference must complete within a latency budget sufficient to maintain gimbal tracking
- Inference process must be isolated from storage, telemetry, and comms tasks
- Daily downlink cap: 1 GB. Daily uplink cap: 100 MB.
- Communications: weekdays only, CCSDS packet protocol
- Downlink priority: health telemetry > science data > compressed imagery > raw imagery

**This codebase is Python first, Rust second.** All Python code must be written in a
Rust-idiomatic style so that translation to Rust is mechanical: strong typing everywhere,
`@dataclass(frozen=True)` for all data-carrying structs, `enum.Enum` for all enumerations,
no dynamic dispatch, no duck typing, explicit `Optional[T]` instead of `None` returns, and
typed message-passing between processes via `multiprocessing.Queue`.

---

## Inter-Process Communication Topology

The process topology below shows all named queues and the direction of data flow.
Preprocessing runs *inside* the inference process (same process, function call) to keep the
hot-path latency tight and avoid serialisation overhead on the critical inference path.

```
imaging_process        --[raw_frame_queue]-->      inference_process
                                                   (preprocessing runs here)
inference_process      --[inference_queue]-->      controller_process
controller_process     --[gimbal_queue]-->          (hardware gimbal driver stub)
controller_process     --[telemetry_queue]-->       telemetry_process
inference_process      --[storage_queue]-->         storage_process
storage_process        --[downlink_queue]-->        comms_process
telemetry_process      --[downlink_queue]-->        comms_process
comms_process          --[uplink_queue]-->          ops/main.py (model deployment)
any subsystem          --[fault_queue]-->           fault_process
any subsystem          --[heartbeat_queue]-->       fault_process
fault_process          --[mode_queue]-->            ops/main.py
```

All queue payloads are frozen dataclasses defined in `src/pact/types/messages.py`.
Every message has a `msg_type: MessageType` discriminant field.

---

## Dependency Layer Order

The import graph is strictly layered. Lower layers must never import from higher layers.

```
types                                          ← dependency root; no internal imports
  ↓
model / preprocessing / imaging                ← import from types only
  ↓
controller                                     ← imports from types + preprocessing
  ↓
storage / telemetry / comms                    ← import from types + model (inference engine)
  ↓
fault                                          ← imports from types + all subsystems' message types
  ↓
ops                                            ← imports everything; orchestration layer
```

Violation of this order is a circular-import error and must be treated as a build-breaking bug.

---

## Coding Conventions Summary (§3, Non-Negotiable)

### 3.1 Types
- Every function has complete type annotations. No bare `Any`.
- `@dataclass(frozen=True)` for all structs. No mutable dataclasses unless commented.
- `Optional[T]` explicitly. Never return `None` from a function typed as returning `T`.
- `Final[T]` for constants. `TypeAlias` for complex repeated types.
- Numpy arrays: annotate dtype and shape in comments — `# np.ndarray[float32, (H, W, C)]`
- No `**kwargs`. No `*args` except in test helpers.

### 3.2 Enums
- `enum.Enum` (not `IntEnum` unless the value must be serialized as an int).
- String values mirror the member name for readability in logs: `IDLE = "IDLE"`.

### 3.3 Error Handling
- Library code returns `Result[T, E]` — never raises exceptions.
- `Result = Union[Ok[T], Err[E]]` — both are frozen dataclasses.
- Scripts and process entry points may raise exceptions for unrecoverable startup failures.

### 3.4 Concurrency
- `multiprocessing.Process` + `multiprocessing.Queue` — CPU-bound (inference, preprocessing).
- `threading.Thread` + `queue.Queue` — I/O-bound (camera, comms, storage).
- `asyncio` — I/O-multiplexed (comms scheduler, uplink chunk assembler).
- No shared mutable state across any concurrency boundary.
- Inference subsystem **must** use `multiprocessing.Process` (REQ-AIML-COMP-002).

### 3.5 Inter-Subsystem Messaging
- All cross-boundary values are frozen dataclasses from `pact/types/messages.py`.
- Every message has `msg_type: MessageType` as its discriminant (first field).

### 3.6 Logging
- `structlog` everywhere. JSON renderer for flight; console renderer for development.
- Every log entry: `subsystem` (str) + `event` (snake_case str) + structured fields.

### 3.6 Configuration
- All parameters live in `config/default.toml`. No magic numbers in source code.
- `ops/config_loader.py` loads TOML once at startup into typed `PactConfig` dataclasses.
- No subsystem reads TOML directly.

### 3.7 General
- **Line length: 100 characters.**
- Imports grouped at top: stdlib → third-party → internal.
- No circular imports. Layered dependency graph (see above).
- Every module has a module-level docstring with purpose and requirement IDs.

---

## Configuration Reference

All tunable thresholds, timeouts, and parameters are defined in:

```
config/default.toml    ← development defaults (source of truth)
config/flight.toml     ← flight overrides merged on top of default at load time
```

The `[controller]`, `[inference]`, `[comms]`, `[storage]`, and `[fault]` sections map
directly to the dataclasses in `src/pact/types/config.py`.

---

## Subsystem CLAUDE.md Locations

| Subsystem       | Context file                              |
|-----------------|-------------------------------------------|
| types           | `src/pact/types/CLAUDE.md`               |
| model           | `src/pact/model/CLAUDE.md`               |
| preprocessing   | `src/pact/preprocessing/CLAUDE.md`       |
| controller      | `src/pact/controller/CLAUDE.md`          |
| imaging         | `src/pact/imaging/CLAUDE.md`             |
| comms           | `src/pact/comms/CLAUDE.md`              |
| storage         | `src/pact/storage/CLAUDE.md`            |
| telemetry       | `src/pact/telemetry/CLAUDE.md`          |
| fault           | `src/pact/fault/CLAUDE.md`              |
| ops             | `src/pact/ops/CLAUDE.md`               |
