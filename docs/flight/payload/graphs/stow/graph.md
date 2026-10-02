# flight.payload.graphs.stow.graph

**Source:** `packages/flight/src/flight/payload/graphs/stow/graph.py`
**Kind:** pure module

## Purpose

The STOW graph orchestrator: spec, edges, initial state, and one-tick step.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `EDGES` | tuple | The `MOVING -> HELD` `VERIFIED_STABLE` edge |
| `spec` | function | Two-node `GraphSpec` |
| `initial_state` | function | MOVING entered at the tick's time |
| `step` | function | Advances the graph one tick |

## Inputs and outputs

`spec(params) -> GraphSpec[StowNode]`, `initial_state(inputs, params) -> State`,
and `step(state, inputs, params) -> (State, GraphOutcome[StowNode])`.

## Behavior

1. Activation mismatch and containment inhibit; containment requests SAFE.
2. MOVING inhibits on stale feedback or the latched timeout.
3. `stow_complete` plus `health.inhibit_confirmed` and fresh feedback commits
   the single `VERIFIED_STABLE` edge to HELD.
4. Elapsed time past `xeryon.stow_timeout_s` latches `timeout_latched`,
   inhibits, emits `GIMBAL_SAFETY_TIMEOUT`, and requests SAFE once.

## Errors and faults

`GIMBAL_SAFETY_TIMEOUT` on the latched timeout; SAFE intent on containment or
timeout.

## Messages

None.

## Configuration

Reads `PactConfig` through `GraphParameters`.

## Constraints

Pure; at most one edge commits per tick.

## Related documents

- [`flight.payload.graphs.stow`](../stow.md)
- [`flight.payload.graphs.base`](../base.md)
