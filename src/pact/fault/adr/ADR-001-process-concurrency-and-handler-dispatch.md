# ADR-001: multiprocessing.Process + Per-FaultCode Handler Dispatch Table

**Status:** Accepted
**Date:** 2026-04-03
**Req IDs:** REQ-SAFE-HIGH-002, REQ-GIMB-HIGH-003, GOAL-006

## Context

The fault subsystem has two responsibilities:
1. **Watchdog** — monitor heartbeats from all other processes; declare a process dead if its
   heartbeat misses `watchdog_max_miss_count` intervals.
2. **Fault handler** — receive `FaultEventMsg` values from any subsystem, invoke the
   appropriate handler, and (if necessary) emit a `ModeChangeMsg` to trigger safe mode.

The fault process must remain alive and responsive even when other processes are stalled or
dead. If it shared a GIL with the processes it monitors, a GIL contention event in a misbehaving
process could delay fault detection. Additionally, the watchdog checks heartbeats on a timer;
`time.sleep()` in a thread is subject to GIL starvation.

For the handler dispatch, two patterns were considered:
1. **`if/elif` chain** — readable but O(n) and requires modifying the dispatch logic to add
   new fault codes.
2. **`FAULT_HANDLERS` dict** — O(1) lookup, exhaustive at definition time, and maps directly
   to a Rust `match` arm per `FaultCode` variant.

## Decision

1. **Concurrency: `multiprocessing.Process`** — the fault process runs in its own OS process
   with its own GIL. The watchdog timer (`time.sleep(config.watchdog_interval_s)`) runs
   without GIL interference from misbehaving subsystems. Cross-process communication uses
   `multiprocessing.Queue` for `HeartbeatMsg`, `FaultEventMsg`, and `ModeChangeMsg`.

2. **Handler dispatch: `FAULT_HANDLERS` dict** — a `Final[dict[FaultCode, Callable[...]]]`
   mapping every `FaultCode` to a handler function. `process.py` calls
   `FAULT_HANDLERS[fault_event.fault_code](fault_event)` and dispatches the returned
   `Optional[ModeChangeMsg]`. At startup, an assertion verifies that every `FaultCode` member
   has an entry in `FAULT_HANDLERS` (exhaustiveness check).

## Consequences

### Positive
- Fault detection is immune to GIL starvation in monitored processes.
- Adding a new `FaultCode` enum member requires adding a handler to `FAULT_HANDLERS` and
  the startup assertion will catch omissions immediately at runtime (before any fault occurs).
- Handler functions are individually unit-testable: pass a `FaultEventMsg`, assert on
  the returned `Optional[ModeChangeMsg]`.
- Maps cleanly to Rust: `FAULT_HANDLERS` becomes a `match fault_code { ... }` expression.

### Negative / Trade-offs
- `multiprocessing.Process` adds serialization overhead to all `HeartbeatMsg` values, which
  are high-frequency (one per subsystem per `watchdog_interval_s`). `HeartbeatMsg` is small
  (subsystem name + sequence int) so pickle overhead is negligible.
- If the fault process itself crashes, there is no watchdog for the watchdog. Mitigation:
  `ops/main.py` monitors all spawned processes and restarts the fault process if it dies
  (restart is itself a fault event, logged to structlog before restart).
- `FAULT_HANDLERS` must cover all `FaultCode` members including `FaultCode.NONE`. The `NONE`
  handler is a no-op that returns `None`. This is explicit and intentional.
