# flight.payload.graphs.safe

**Source:** `packages/flight/src/flight/payload/graphs/safe.py`
**Kind:** pure module

## Purpose

The SAFE graph holds one `INHIBITED` node that always emits an
`InhibitReference` regardless of feedback. SAFE is independent motion
inhibition: the graph holds no pose, issues no stow, and waits for external
graph selection.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SafeNode` | enum | `INHIBITED` |
| `State` | dataclass | Activation key and node |
| `spec` | function | Single-node `GraphSpec` |
| `initial_state` | function | Cold INHIBITED state |
| `step` | function | Always the inhibit reference with the off policy |

## Inputs and outputs

`spec(params) -> GraphSpec[SafeNode]`, `initial_state(inputs, params) -> State`,
and `step(state, inputs, params) -> (State, GraphOutcome[SafeNode])`.

## Behavior

Every tick returns the same state and an inhibit outcome; SAFE never poses
or tracks and never writes a CoG.

## Errors and faults

None.

## Messages

None.

## Configuration

Reads `PactConfig` through `GraphParameters`.

## Constraints

Pure; no `SystemMode`, no motion references, no imaging.

## Related documents

- [`flight.payload.graphs`](../graphs.md)
- [`flight.payload.graphs.base`](base.md)
