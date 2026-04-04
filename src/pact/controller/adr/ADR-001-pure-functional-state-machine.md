# ADR-001: Pure Functional Arbiter State Machine

**Status:** Accepted
**Date:** 2026-04-03
**Req IDs:** REQ-AIML-GIMB-001, REQ-AIML-GIMB-008, REQ-GIMB-HIGH-001, REQ-GIMB-HIGH-004

## Context

The gimbal safety arbiter is the most safety-critical component in PACT. It controls physical
hardware (the gimbal) and must never issue a command outside of its defined safety envelopes
(confidence gate, min area gate, deadband, rate limit). Errors in arbiter logic could cause
gimbal runaway, a safety-critical fault (`FaultCode.GIMBAL_RUNAWAY`).

The controller process must also log every state transition as a `TelemetryEventMsg` for
ground-based audit. Two design patterns exist for a state machine with side effects:

1. **Stateful class with queues** — `GimbalArbiter` holds references to output queues and
   puts messages directly on them during `step()`.
2. **Pure function** — `GimbalArbiter.step()` takes state in, returns new state + output
   messages. The surrounding process loop dispatches the outputs to the appropriate queues.

## Decision

`GimbalArbiter.step()` is a **pure function**:

```
(ArbiterState, InferenceResultMsg, float) → (ArbiterState, Optional[GimbalCommandMsg], list[TelemetryEventMsg])
```

- `GimbalArbiter` holds no queue references and has no side effects.
- `ArbiterState` is an immutable frozen dataclass.
- All state transitions are returned as data; `controller/process.py` dispatches them.
- The full safety pipeline (confidence gate → min area → blob tracker → EMA filter →
  deadband → rate limit → arbiter) runs as a sequence of pure function calls in `process.py`
  before `arbiter.step()` is invoked.

The concurrency primitive for the controller process is `multiprocessing.Process` with
`multiprocessing.Queue` for all inter-process communication.

## Consequences

### Positive
- Pure functions are trivially unit-testable: no mocks needed for queues. Pass in state and
  result, assert on returned state and commands.
- State machine correctness can be verified exhaustively: enumerate all `(GimbalState, input)`
  combinations and assert the transition table.
- No shared mutable state: `ArbiterState` crossing a function boundary is always a fresh
  frozen copy. Impossible to accidentally mutate state mid-step.
- Mirrors Rust's ownership model: `step()` consumes the old state and produces a new one,
  which translates mechanically to Rust's `match` on an owned enum.
- Easier to replay telemetry: given a log of `InferenceResultMsg` values, the arbiter can
  be re-run deterministically to reconstruct the state history.

### Negative / Trade-offs
- `list[TelemetryEventMsg]` returned from `step()` must be dispatched by `process.py`. If
  `process.py` crashes between `step()` returning and dispatching, telemetry is lost. This
  is acceptable — the watchdog will detect the crash.
- `process.py` becomes the integration point for the full pipeline. Its correctness cannot
  be unit-tested without spinning up a real process; this is covered by
  `tests/integration/test_controller_pipeline.py`.
