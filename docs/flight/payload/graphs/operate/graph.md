# flight.payload.graphs.operate.graph

**Source:** `packages/flight/src/flight/payload/graphs/operate/graph.py`
**Kind:** pure module

## Purpose

The OPERATE graph orchestrator: spec, edges, initial state, one-tick step, and
`apply_command` for routed commands.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `EDGES` | tuple | All automatic and directed command edges |
| `spec` | function | Four-node `GraphSpec` |
| `initial_state` | function | Cold TRACKING with a residual history seeded from fresh feedback |
| `step` | function | Advances the graph one tick |
| `apply_command` | function | `CommandOutcome` for declared directed command edges |

## Inputs and outputs

`spec(params) -> GraphSpec[OperateNode]`, `initial_state(inputs, params) -> State`,
`step(state, inputs, params) -> (State, GraphOutcome[OperateNode])`, and
`apply_command(state, command, inputs, params)` returns
`Result[CommandOutcome[State, OperateNode], FaultCode]`.

## Behavior

1. Activation mismatch, containment, stale feedback, and flagged vision emit
   inhibit (containment and flags with a SAFE intent).
2. A valid routed command commits before automatic edges; guard failures
   return `Err(COMMAND_INVALID)` with the state untouched.
3. Accepted vision commits `VISION_ACQUIRED` before limb arrival and the
   rewind timer; bounded coast exhaustion commits `COAST_EXHAUSTED` to REWIND
   away from the limb or HOLD at the limb, exactly once via `loss_handled`.
4. On each committed edge the destination node's reference and policy apply
   the same tick and no second transition fires.

## Errors and faults

`INFERENCE_NAN` on flagged vision; `COMMAND_INVALID` on failed guards.

## Messages

None.

## Configuration

Reads `PactConfig` through `GraphParameters`.

## Constraints

Pure; at most one edge per tick; manual HOLD never auto-exits.

## Related documents

- [`flight.payload.graphs.operate`](../operate.md)
- [`flight.payload.graphs.base`](../base.md)
