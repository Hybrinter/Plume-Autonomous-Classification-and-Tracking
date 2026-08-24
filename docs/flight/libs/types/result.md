# flight.libs.types.result

**Source:** `packages/flight/src/flight/libs/types/result.py`
**Kind:** pure module

## Purpose

The module defines `Ok`, `Err`, and `Result[T, E]`. Library code returns these types for
recoverable failures. Callers pattern-match on success or error.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `Ok` | class | Success wrapper holding `value: T` |
| `Err` | class | Error wrapper holding `error: E` |
| `Result` | type alias | `Union[Ok[T], Err[E]]` |

## Inputs and outputs

| Entry point | Inputs | Outputs |
| --- | --- | --- |
| `Ok(value)` | Success value `T` | Frozen `Ok[T]` instance |
| `Err(error)` | Error value `E` | Frozen `Err[E]` instance |

## Behavior

1. A function returns `Ok(value)` on success.
2. A function returns `Err(error)` on a recoverable failure.
3. The caller checks `isinstance(result, Ok)` or `isinstance(result, Err)` before reading
   `.value` or `.error`.

## Errors and faults

None. This module defines the error carrier types only.

## Messages

None.

## Configuration

None.

## Constraints

- `Ok` and `Err` are `@dataclass(frozen=True)` without slots.
- The explicit `Generic[T]` and `Union` forms are the stable public contract.
- Callers must not read `.value` without an `Ok` check first.
- Process entry points may raise only for unrecoverable startup failures.

## Related documents

- [`flight.libs.types`](../types.md)
- [`flight.libs.types.enums`](enums.md)
