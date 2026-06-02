# ADR 0005: Pure-core + thin-shell apps; `Result` over exceptions

**Status:** Accepted (2026-05-30)

## Context

Control, tracking, and FDIR logic must be deterministic, testable without mocks, and replayable
from logs. Mixing that logic with I/O (bus, clock, drivers) makes it hard to test and reason about
under concurrency. Error handling must be explicit at the boundaries where a caller can react.

## Decision

Split every app into a **pure core** and a **thin shell**:

- **Pure cores** (e.g. `PayloadController.step`, `GimbalArbiter.step`, the tracking estimators, the
  FDIR `check_heartbeats`/`decide_mode_change`) take state + inputs (including `now` as a plain
  float) and return new state + output messages. No I/O, no bus, no clock reads, no logging. State
  is threaded in and out.
- **Thin shells** (the `*App` classes) own the bus subscriptions, the injected `Clock` and drivers,
  the loop, and message construction; they call the pure cores.

Library code returns **`Result[T, E]`** (`Ok` | `Err`) rather than raising; only process entry
points raise, and only for unrecoverable startup failures. Never read `.value` without an
`Ok`/`Err` check.

## Consequences

- Cores are unit-tested directly with plain values and a `ManualClock`; no process spin-up or
  mocking. The SIL drives the same cores deterministically.
- Time is injected, so monotonic intervals/rate-limits and wall-clock stamps are controllable and
  the cores stay deterministic.
- The `Result` discipline forces faults to be handled at the boundary (emit a `FaultEventMsg`,
  degrade) instead of propagating as exceptions across the bus.
