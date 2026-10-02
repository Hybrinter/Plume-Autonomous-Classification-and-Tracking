# flight.payload.graphs.idle

**Source:** `packages/flight/src/flight/payload/graphs/idle.py`
**Kind:** pure module

## Purpose

The IDLE graph holds one `HOLD` node. It captures the entry pose from the
first fresh encoder sample and holds it through the pose envelope. Without
fresh feedback it inhibits and keeps no target. No imaging or inference runs
in IDLE.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `IdleNode` | enum | `HOLD` |
| `State` | dataclass | Activation key, node, captured `target_rad` |
| `spec` | function | Single-node `GraphSpec` |
| `initial_state` | function | Cold HOLD state; pose captured when feedback is fresh |
| `step` | function | PoseReference hold or inhibit outcome |

## Inputs and outputs

`spec(params) -> GraphSpec[IdleNode]`, `initial_state(inputs, params) -> State`,
and `step(state, inputs, params) -> (State, GraphOutcome[IdleNode])`.

## Behavior

1. `initial_state` captures the encoder angle when feedback is fresh.
2. `step` inhibits on activation mismatch, containment (with a SAFE
   intent), or missing/stale feedback.
3. The first later fresh sample sets `target_rad`; thereafter the pose holds.

## Errors and faults

None directly; containment surfaces a `SystemRequestIntent.SAFE`.

## Messages

None.

## Configuration

Reads `PactConfig` through `GraphParameters`.

## Constraints

Pure; no `SystemMode`, no edges, no imaging.

## Related documents

- [`flight.payload.graphs`](../graphs.md)
- [`flight.payload.graphs.base`](base.md)
