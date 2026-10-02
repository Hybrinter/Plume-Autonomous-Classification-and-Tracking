# flight.payload.graphs.operate.state

**Source:** `packages/flight/src/flight/payload/graphs/operate/state.py`
**Kind:** pure module

## Purpose

The OPERATE state record, node vocabulary, and shared ingest/bookkeeping
helpers used by every node step.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `OperateNode` | enum | `TRACKING`, `REWIND`, `FAST_REWIND`, `HOLD` |
| `HoldReason` | enum | `LIMB_WAIT` or `MANUAL` |
| `TargetState` | dataclass | Stored CoG and last scene-rate terms |
| `HoldState` | dataclass | Hold reason and captured target |
| `State` | dataclass | Activation key, node, ancestry, liveness, residual, target, hold, seen window |
| `vision_window_s` | function | Acceptance window: max(observation age, residual history horizon) |
| `accept_vision` | function | Context/window/dedup gates plus blob matching and centroid/pinhole fill |
| `bookkeep_vision` | function | Apply one accepted sample to misses, liveness, and the seen window |
| `acquire_resets_residual` | function | TRACKING acquire policy needing a cold residual |

## Inputs and outputs

Data and helpers only: `State` carries the activation key, node, tracked
ancestry, liveness, residual state and history, target, hold, seen-vision
window, `last_rate_decision`, `last_e_az`, and `vision_disposition`.
Signatures: `vision_window_s(params) -> float`,
`accept_vision(state, inputs, params)`,
`bookkeep_vision(state, vision, inputs, params) -> State`, and
`acquire_resets_residual(previous_node, previous_aggregate_live,
previous_blob_ids, new_blob_ids) -> bool`.

## Behavior

`accept_vision` applies the context, shutter-window, and duplicate gates, runs the confidence/area gates and blob matching, and fills the union centroid and pinhole error. `bookkeep_vision` applies one accepted sample to misses, liveness, and the seen window.

## Errors and faults

None directly.

## Messages

None.

## Configuration

Reads `PactConfig` through `GraphParameters`.

## Constraints

Pure data and helpers; frozen slots.

## Related documents

- [`flight.payload.graphs.operate`](../operate.md)
