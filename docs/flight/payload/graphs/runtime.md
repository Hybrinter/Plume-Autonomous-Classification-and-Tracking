# flight.payload.graphs.runtime

**Source:** `packages/flight/src/flight/payload/graphs/runtime.py`
**Kind:** pure module

## Purpose

The runtime is the closed-union dispatch across the five pure graphs. It is
not an engine: `step` and `apply_command` are explicit `match` dispatch over
the concrete graph-state union, and `initial_state` builds the cold state for
the caller's authoritative graph selection. No activation bookkeeping, epoch
generation, or graph selection lives here.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `GraphState` | union | `idle.State | safe.State | stow.State | init.State | operate.State` |
| `GraphOutcomeUnion` | union | `GraphOutcome` over each graph's node enum |
| `GraphSpecUnion` | union | `GraphSpec` over each graph's node enum |
| `initial_state` | function | Cold state for the selected `GraphId` |
| `spec` | function | Declared spec (topology + default policy) for one `GraphId` |
| `entry_policy` | function | Policy in force at activation; OPERATE resolves the TRACKING node override |
| `step` | function | One tick of whichever graph owns the state |
| `apply_command` | function | OPERATE command application only; other graphs `Err(COMMAND_INVALID)` |

## Inputs and outputs

`initial_state(graph_id, inputs, params) -> GraphState`;
`spec(graph_id, params) -> GraphSpecUnion`;
`entry_policy(graph_id, params) -> Result[EffectivePolicy, FaultCode]`;
`step(state, inputs, params) -> (GraphState, GraphOutcomeUnion)`;
`apply_command(state, command, inputs, params)` returns
`Result[CommandOutcome[operate.State, OperateNode], FaultCode]`.

## Behavior

1. `initial_state` matches `GraphId` to the per-graph builder.
2. `step` matches the concrete state type to the per-graph step; every
   graph step already returns its typed `GraphOutcome`, so the dispatch
   returns it directly.
3. `apply_command` delegates to `operate.apply_command`; every other graph
   state returns `Err(COMMAND_INVALID)`.
4. `entry_policy` resolves OPERATE to `operating_policy(tracking)` so the
   first capture under a fresh activation already carries the entry node's
   configured override; other graphs return their declared spec policy.

## Errors and faults

`Err(COMMAND_INVALID)` for commands against non-OPERATE graphs.

## Messages

None.

## Configuration

Reads `PactConfig` through `GraphParameters`.

## Constraints

Pure dispatch only. The shell (`PayloadApp`) owns activation mapping, epochs,
and graph selection; nothing here selects graphs or owns epochs.

## Related documents

- [`flight.payload.graphs`](../graphs.md)
- [`flight.payload.graphs.base`](base.md)
