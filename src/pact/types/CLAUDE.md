# PACT Types Package — Subsystem Context

## Purpose

Foundation types package. All other PACT subsystems import from here. Defines the complete
shared vocabulary of enumerations, inter-process message dataclasses, and typed configuration
dataclasses used across the entire system.

## Satisfies

- REQ-AIML-COMP-001 — type safety foundation: all cross-boundary values are frozen dataclasses
  or enums; no untyped dicts or bare `Any` values cross subsystem boundaries.
- REQ-AIML-COMP-002 — process isolation contract: typed messages are the only legal channel
  between processes; enforcement is structural (frozen dataclasses cannot be mutated).

## Owns (Message Types Produced)

None. This package defines all message schemas but produces no messages at runtime. It is a
pure type library with no process or queue logic.

## Consumes

Nothing. This package imports only Python stdlib (`enum`, `dataclasses`, `typing`). No other
PACT submodule is imported here, making it the dependency root.

## Key Invariants

1. **No circular imports.** No other pact submodule (`pact.model`, `pact.controller`, etc.)
   is imported anywhere in this package. Violations must be treated as build-breaking bugs.

2. **All types are frozen.** Every data-carrying struct is `@dataclass(frozen=True)`. Every
   enumeration uses `enum.Enum`. No mutable state is defined here.

3. **Enums, not strings, for all discriminants.** Every `msg_type` field uses `MessageType`,
   not a raw string. Every `fault_code` field uses `FaultCode`, etc.

4. **Result[T, E] mirrors Rust.** `Ok[T]` and `Err[E]` are frozen dataclasses.
   Library code must not raise exceptions — return `Err(...)` instead.

5. **Config defaults match default.toml exactly.** The default field values in `config.py`
   must always be kept in sync with `config/default.toml`. A CI check should enforce this.

## Known Gaps / TODOs

- `UploadChunkMsg` was not in the original §4.2 specification but is required by
  `comms/uplink.py`'s `process_uplink_chunk()` function. It has been added to `messages.py`
  with the fields: `msg_type`, `timestamp_utc`, `chunk_index: int`, `total_chunks: int`,
  `data: bytes`, `expected_crc32: int`. The `MessageType.UPLINK_CHUNK` enum member was also
  added to `enums.py` accordingly.

- `RawFrameMsg.raw_bands` and similar numpy array fields are typed as `object` at runtime
  because frozen dataclasses cannot hold numpy arrays with `np.ndarray` type annotations
  enforced at construction time. Annotate dtype and shape in comments; enforce shapes in
  the producing subsystem before putting the message on the queue.

- No schema versioning yet. If message schemas change between software versions during a
  multi-day mission, queued messages may be incompatible. Add a `schema_version: int` field
  to `RawFrameMsg` and `InferenceResultMsg` before first flight integration test.

- `DownlinkPriority` uses `IntEnum`-style integer values (`HEALTH_TELEMETRY = 0`, etc.) but
  is declared as `enum.Enum` (not `IntEnum`). This is intentional — the integer value is used
  internally by `queue.PriorityQueue` but must not be serialised as a bare int in CCSDS
  packets. If direct int comparisons are needed, access `.value` explicitly.
