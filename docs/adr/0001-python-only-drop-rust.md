# ADR 0001: Python-only; drop the Rust migration

**Status:** Accepted (2026-05-30)

## Context

The original plan was to prototype in Python and then mechanically translate to Rust, so the
codebase was written "Rust-idiomatically" (frozen dataclasses, enum discriminants, `Result[T,E]`,
no dynamic dispatch) and carried a Rust-Migration Contract. In practice the flight workload is
dominated by libraries that are already thin wrappers over compiled code (numpy/scipy, the model
runtime), so the language of the orchestration layer is not the performance bottleneck. The Rust
rewrite added a large, ongoing translation tax for little runtime benefit on an ISS-attached
payload that is ground-recoverable.

## Decision

Keep the system **Python-only**. Drop the Rust migration and the Rust-Migration Contract. Retain
the disciplines that are good engineering on their own merits -- frozen dataclasses, `Result[T,E]`
error handling, strong typing, `enum` value==name -- but stop treating "mechanical translatability
to Rust" as a constraint. Relax "no dynamic dispatch / no duck typing" to **allow statically-typed
`Protocol` interfaces** (the basis of the HAL).

## Consequences

- The HAL becomes clean `Protocol`-based dispatch instead of an enum-of-drivers workaround.
- One language, one toolchain, one test suite; no FFI boundary or dual maintenance.
- Heavy/native acceleration, if ever needed on the hot path, is reachable via targeted native
  extensions rather than a wholesale rewrite.
