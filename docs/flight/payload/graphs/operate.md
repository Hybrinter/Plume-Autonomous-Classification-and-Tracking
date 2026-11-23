# flight.payload.graphs.operate

**Source:** `packages/flight/src/flight/payload/graphs/operate`
**Kind:** package

## Purpose

The OPERATE graph tracks a plume CoG with the residual estimator, hunts back
toward the science limb on bounded coast loss, promotes the hunt to the
hardware rate on a timer, and holds either limb-wait or a manual pose.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`state`](operate/state.md) | pure module | `OperateNode`, `State`, ingest and bookkeeping helpers |
| [`tracking`](operate/tracking.md) | pure module | TRACKING node step (scene + residual rate law) |
| [`rewind`](operate/rewind.md) | pure module | REWIND node step (smear-cap hunt) |
| [`fast_rewind`](operate/fast_rewind.md) | pure module | FAST_REWIND node step (hardware-rate hunt) |
| [`hold`](operate/hold.md) | pure module | HOLD node step (pose or inhibit) |
| [`graph`](operate/graph.md) | pure module | Spec, edges, initial state, step, and `apply_command` |

## Package interface

Re-exports `State`, `OperateNode`, `HoldReason`, `spec`, `initial_state`,
`step`, and `apply_command`.

## Interactions

Commands are validated against the declared directed edges and committed
before automatic edges; accepted vision outranks limb arrival and the rewind
timer. Only the OPERATE graph applies routed commands.

## Constraints

Pure; no `SystemMode`; vision ingest rejects stale keys, policy revisions,
future/expired shutters, and duplicate frame ids.

## Related documents

- [`flight.payload.graphs`](../graphs.md)
- [`flight.payload.graphs.base`](base.md)
