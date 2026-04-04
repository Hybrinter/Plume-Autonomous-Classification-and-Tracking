# ADR-001: Rust-Idiomatic Type System

**Status:** Accepted
**Date:** 2026-04-03
**Req IDs:** REQ-AIML-COMP-001, REQ-AIML-COMP-002 (process isolation), general coding standard

## Context

PACT is Python-first but must translate mechanically to Rust in a future phase. The type system
design must ensure that Python code can be ported to Rust without architectural rework. Python's
permissive dynamic typing, mutable defaults, and duck typing are the primary risks to this goal.

Additionally, all data passed across concurrency boundaries must be immutable and serializable
to prevent subtle shared-state bugs in a multi-process system.

## Decision

Adopt the following type system conventions across all subsystems:

- **Frozen dataclasses** — all data-carrying structs use `@dataclass(frozen=True)`. No mutable
  dataclasses unless a comment explains why mutation is required and documents the exception.
- **Result type** — functions that can fail return `Result[T, E]` defined as
  `Union[Ok[T], Err[E]]` where `Ok` and `Err` are frozen dataclasses. This mirrors Rust's
  `Result<T, E>`. No exceptions are raised in library code; only scripts and process entry
  points may raise.
- **MessageType discriminant** — every inter-process message has a `msg_type: MessageType`
  field as its first field. `MessageType` is a `str`-valued `enum.Enum`. This mirrors Rust's
  enum variant discriminant and enables exhaustive pattern matching in future Rust code.
- **No bare `Any`** — all function parameters and return values have complete type annotations.
  `Optional[T]` is used explicitly; no implicit `None` returns from typed functions.
- **`Final[T]`** for constants, **`TypeAlias`** for complex repeated types.
- **No `**kwargs`, no `*args`** except in test helpers.

## Consequences

### Positive
- Python code structure maps 1:1 to Rust structs and enums with no architectural rework.
- Frozen dataclasses are safe to pass across `multiprocessing.Queue` without deep-copy guards.
- `Result` type makes all failure modes explicit at the call site, eliminating silent `None`
  propagation.
- `MessageType` discriminant enables exhaustive type-narrowing in both Python (via `match`)
  and future Rust (via `match` on enum variants).

### Negative / Trade-offs
- More verbose than idiomatic Python. Reviewers unfamiliar with Rust-idiomatic Python may find
  the style unusual.
- Frozen dataclasses holding `np.ndarray` fields are technically mutable (the array contents
  can be modified). This is documented as an accepted exception in the `model/` subsystem for
  `InferenceEngine`, which holds a `torch.nn.Module`.
- `tomllib` (stdlib in Python 3.11+) cannot deserialize directly into frozen dataclasses;
  `config_loader.py` must perform explicit field mapping.
