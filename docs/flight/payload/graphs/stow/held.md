# flight.payload.graphs.stow.held

**Source:** `packages/flight/src/flight/payload/graphs/stow/held.py`
**Kind:** pure module

## Purpose

The HELD node emits only an inhibit reference; the bounded move already
completed and the actuator stays inhibited until external graph selection.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `step` | function | `InhibitReference` outcome under the off policy |

## Inputs and outputs

`step(state, inputs, params) -> (State, NodeOutcome[StowNode])`.

## Behavior

`step` returns the state unchanged with an `InhibitReference` under the
off policy; the bounded move already completed upstream.

## Errors and faults

None directly.

## Messages

None.

## Configuration

Reads `PactConfig` through `GraphParameters`.

## Constraints

Pure.

## Related documents

- [`flight.payload.graphs.stow`](../stow.md)
