# flight.payload.graphs.operate.hold

**Source:** `packages/flight/src/flight/payload/graphs/operate/hold.py`
**Kind:** pure module

## Purpose

The HOLD node emits a `PoseReference` to a captured or commanded elevation
under the pose envelope. Limb-wait entries capture the fresh encoder angle on
arrival; without a target and without fresh feedback the node inhibits.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `step` | function | Updated state plus a `PoseReference` or inhibit outcome |

## Inputs and outputs

`step(state, inputs, params) -> (State, NodeOutcome[OperateNode])`.

## Behavior

`step` emits a `PoseReference` to `hold.target_rad` under the pose
envelope when a target exists and feedback is fresh; otherwise it
inhibits. `last_rate_decision` clears to `None` while holding.

## Errors and faults

None directly.

## Messages

None.

## Configuration

Reads `PactConfig` through `GraphParameters`.

## Constraints

Pure; the residual is not fed while holding.

## Related documents

- [`flight.payload.graphs.operate`](../operate.md)
