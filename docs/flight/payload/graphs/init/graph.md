# flight.payload.graphs.init.graph

**Source:** `packages/flight/src/flight/payload/graphs/init/graph.py`
**Kind:** pure module

## Purpose

The INIT graph orchestrator: spec, edges, initial state, and one-tick step
over the effect chain.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `EDGES` | tuple | The `SELFTEST -> MODEL_LOAD -> HOME -> READY` `EFFECT_COMPLETED` chain |
| `spec` | function | Four-node `GraphSpec` |
| `initial_state` | function | Cold SELFTEST with the configured home target |
| `step` | function | Advances the graph one tick |

## Inputs and outputs

`spec(params) -> GraphSpec[InitNode]`, `initial_state(inputs, params) -> State`,
and `step(state, inputs, params) -> (State, GraphOutcome[InitNode])`.

## Behavior

1. Activation mismatch and containment inhibit; containment requests SAFE.
2. Each node issues its effect once; results apply only when key, kind, and
   id match an already-issued intent. PENDING and unsolicited results are
   ignored; FAILED latches `failed` with one fault and one SAFE request.
3. SUCCEEDED results commit one `EFFECT_COMPLETED` edge; HOME additionally
   requires nonempty arrival evidence.
4. READY waits on `InitVerificationResult`; a VERIFIED result with nonempty
   evidence emits one `INIT_COMPLETE` request and stays READY.

## Errors and faults

On effect failure the latched fault is the result's supplied `fault`, or
`GIMBAL_FAULT` when the result reports `NONE`; verification failure raises
`GIMBAL_FAULT`. The SAFE intent emits once on the latched failure.

## Messages

None.

## Configuration

Reads `PactConfig` through `GraphParameters`.

## Constraints

Pure; at most one edge per tick; no default pass on verification.

## Related documents

- [`flight.payload.graphs.init`](../init.md)
- [`flight.payload.graphs.base`](../base.md)
