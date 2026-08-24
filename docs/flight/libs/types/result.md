# flight.libs.types.result

**Source:** `packages/flight/src/flight/libs/types/result.py`
**Kind:** pure module

## Purpose

This module defines explicit success-or-error wrapper types. Library code returns
`Result[T, E]` and does not raise for recoverable failures.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `Ok` | dataclass | Successful result holding `value: T` |
| `Err` | dataclass | Failed result holding `error: E` |
| `Result` | type alias | `Union[Ok[T], Err[E]]` |

## Inputs and outputs

- `Ok(value)` wraps a success value.
- `Err(error)` wraps an error value.
- Callers narrow with `isinstance(result, Ok)` or `isinstance(result, Err)` before reading
  `.value` or `.error`.

## Behavior

1. `Ok` and `Err` are frozen dataclasses.
2. Functions return either `Ok(...)` or `Err(...)`.
3. Callers pattern-match on the wrapper type to handle success and failure paths.

## Errors and faults

This module defines error containers only. Concrete error types (for example `FaultCode`)
come from callers.

## Messages

None.

## Configuration

None.

## Constraints

- Do not read `.value` without an `Ok` check first.
- `Ok` and `Err` use explicit `Generic[T]` / `Generic[E]` forms.
- Process entry points may raise only for unrecoverable startup failures.

## Related documents

- [`flight.libs.types`](flight/libs/types.md)
