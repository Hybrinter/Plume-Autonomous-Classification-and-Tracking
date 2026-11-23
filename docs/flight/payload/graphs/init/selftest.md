# flight.payload.graphs.init.selftest

**Source:** `packages/flight/src/flight/payload/graphs/init/selftest.py`
**Kind:** pure module

## Purpose

The SELFTEST node emits one activation-scoped `SELFTEST` `EffectIntent` and
inhibits while the shell runs the bounded self-test.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `step` | function | Inhibit outcome plus the intent on first emission |

## Inputs and outputs

`step(state, inputs, params) -> (State, NodeOutcome[InitNode])`.

## Behavior

`step` emits the activation-scoped `SELFTEST` `EffectIntent` on the first
call (id `selftest` scoped to the key) and inhibits; later calls return the
state unchanged. It needs no encoder feedback.

## Errors and faults

None directly.

## Messages

None.

## Configuration

Reads `PactConfig` through `GraphParameters`.

## Constraints

Pure; issues the intent once via `state.issued`.

## Related documents

- [`flight.payload.graphs.init`](../init.md)
