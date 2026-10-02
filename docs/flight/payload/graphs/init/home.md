# flight.payload.graphs.init.home

**Source:** `packages/flight/src/flight/payload/graphs/init/home.py`
**Kind:** pure module

## Purpose

The HOME node emits the `HOME` intent once and holds a `PoseReference` to the
configured home elevation under the pose envelope while feedback is fresh.
Without fresh feedback it inhibits; arrival completion requires a SUCCEEDED
result with nonempty evidence.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `step` | function | Pose or inhibit outcome plus the intent on first emission |

## Inputs and outputs

`step(state, inputs, params) -> (State, NodeOutcome[InitNode])`.

## Behavior

`step` waits for fresh feedback before issuing the `HOME` `EffectIntent`
and posing to `home_target_rad` under the pose envelope; stale or missing
feedback inhibits without issuing. Once issued it holds the pose while
feedback stays fresh.

## Errors and faults

None directly.

## Messages

None.

## Configuration

Reads `PactConfig` through `GraphParameters`.

## Constraints

Pure; HOME waits on typed arrival evidence, not target tolerance.

## Related documents

- [`flight.payload.graphs.init`](../init.md)
