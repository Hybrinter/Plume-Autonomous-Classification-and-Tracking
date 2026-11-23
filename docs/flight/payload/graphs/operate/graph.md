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

1. Activation mismatch, containment, and flagged vision inhibit with the
   disabled policy. Containment and flagged vision also emit a SAFE intent.
2. Stale or missing encoder feedback inhibits the motion reference and keeps
   the enabled imaging and inference policy resolved for the current node:
   the operate override applies first, then the live node's `tracking`,
   `rewind`, `fast_rewind`, or `hold` override.
3. A valid routed command commits before automatic edges; guard failures
   return `Err(COMMAND_INVALID)` with the state untouched.
4. Accepted vision commits `VISION_ACQUIRED` before limb arrival and the
   rewind timer. Limb arrival commits `LIMB_ARRIVAL` before `TIMER_EXPIRED`
   when both are true on one tick. Bounded coast exhaustion commits
   `COAST_EXHAUSTED` to REWIND away from the limb or HOLD at the limb, exactly
   once via `loss_handled`.
5. On each committed edge the destination node's reference and policy apply
   the same tick and no second transition fires.
6. A hunt that is still REWIND or FAST_REWIND after `hunt_timeout_s` from
   hunt entry sets `hunt_timeout_latched`, inhibits, and requests SAFE once,
   with `GIMBAL_SAFETY_TIMEOUT`. Later ticks in that hunt inhibit and do not
   request SAFE again. Vision and limb arrival still commit before that
   timeout. `TIMER_EXPIRED` does not.
7. The hunt timeout also applies when encoder feedback is stale or missing.
   It replaces the stale-feedback inhibit and uses the off policy.
8. A valid command still commits in a latched hunt. Each exit from the hunt
   clears `hunt_timeout_latched`.

## Errors and faults

`INFERENCE_NAN` on flagged vision; `COMMAND_INVALID` on failed guards;
`GIMBAL_SAFETY_TIMEOUT` on the hunt timeout.

## Messages

None.

## Configuration

Reads `PactConfig` through `GraphParameters`. The hunt uses
`controller.outer.rewind_sharp_max_s` and `controller.outer.hunt_timeout_s`.

## Constraints

Pure; at most one edge per tick; manual HOLD never auto-exits.

## Related documents

- [`flight.payload.graphs.operate`](../operate.md)
- [`flight.payload.graphs.base`](../base.md)
