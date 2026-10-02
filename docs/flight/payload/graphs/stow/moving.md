# flight.payload.graphs.stow.moving

**Source:** `packages/flight/src/flight/payload/graphs/stow/moving.py`
**Kind:** pure module

## Purpose

The MOVING node emits the configured stow target under the hardware envelope
capped at the bounded stow reference rate, with the configured timeout.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `step` | function | `StowReference` outcome under the off policy |

## Inputs and outputs

`step(state, inputs, params) -> (State, NodeOutcome[StowNode])`.

## Behavior

`step` emits one `StowReference` to `gimbal.stow_el_deg` under the stow
envelope with `xeryon.stow_timeout_s`. Completion and timeout handling live
in the graph.

## Errors and faults

None.

## Messages

None.

## Configuration

Reads `PactConfig` through `GraphParameters`.

## Constraints

Pure; no transition logic here.

## Related documents

- [`flight.payload.graphs.stow`](../stow.md)
