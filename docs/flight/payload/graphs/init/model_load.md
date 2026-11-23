# flight.payload.graphs.init.model_load

**Source:** `packages/flight/src/flight/payload/graphs/init/model_load.py`
**Kind:** pure module

## Purpose

The MODEL_LOAD node emits one activation-scoped `MODEL_LOAD` `EffectIntent`
and inhibits while the shell loads the configured model pair.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `step` | function | Inhibit outcome plus the intent on first emission |

## Inputs and outputs

`step(state, inputs, params) -> (State, NodeOutcome[InitNode])`.

## Behavior

`step` emits the activation-scoped `MODEL_LOAD` `EffectIntent` on the
first call (id `model_load` scoped to the key) and inhibits; later calls
return the state unchanged. It needs no encoder feedback.

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
