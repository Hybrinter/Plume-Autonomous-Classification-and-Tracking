# flight.payload.graphs.operate.rewind

**Source:** `packages/flight/src/flight/payload/graphs/operate/rewind.py`
**Kind:** pure module

## Purpose

The REWIND node hunts toward the science limb at the boresight scene rate plus
the smear cap. At the science maximum the decision is a zero, science-limited
stop.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `step` | function | Updated state plus a `RateReference` or inhibit outcome |

## Inputs and outputs

`step(state, inputs, params) -> (State, NodeOutcome[OperateNode])`.

## Behavior

`step` emits the boresight scene nominal plus the smear cap as a
`RateReference` toward the science limb; at the science maximum the
decision is a zero, science-limited stop. The computed `RateDecision` is
retained on the state.

## Errors and faults

None directly.

## Messages

None.

## Configuration

Reads `PactConfig` through `GraphParameters`.

## Constraints

Pure; stores no target CoG and does not feed the residual.

## Related documents

- [`flight.payload.graphs.operate`](../operate.md)
