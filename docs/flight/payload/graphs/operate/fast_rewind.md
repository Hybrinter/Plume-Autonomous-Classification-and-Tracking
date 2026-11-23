# flight.payload.graphs.operate.fast_rewind

**Source:** `packages/flight/src/flight/payload/graphs/operate/fast_rewind.py`
**Kind:** pure module

## Purpose

The FAST_REWIND node hunts toward the science limb at the hardware slew rate
after the REWIND timer expires; the nominal scene rate is not added.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `step` | function | Updated state plus a `RateReference` or inhibit outcome |

## Inputs and outputs

`step(state, inputs, params) -> (State, NodeOutcome[OperateNode])`.

## Behavior

`step` emits the hardware slew rate as a `RateReference` toward the
science limb without adding the scene nominal. The computed `RateDecision`
is retained on the state.

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
