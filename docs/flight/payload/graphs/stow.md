# flight.payload.graphs.stow

**Source:** `packages/flight/src/flight/payload/graphs/stow`
**Kind:** package

## Purpose

The STOW graph drives one bounded move to the configured stow pose, then holds
inhibited once the control path reports typed completion evidence. A pure
elapsed timeout latches once: inhibit, `GIMBAL_SAFETY_TIMEOUT`, and one SAFE
request.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`state`](stow/state.md) | pure module | `StowNode` and STOW `State` records |
| [`moving`](stow/moving.md) | pure module | MOVING node step (bounded stow reference) |
| [`held`](stow/held.md) | pure module | HELD node step (inhibit) |
| [`graph`](stow/graph.md) | pure module | Spec, edges, initial state, and step |

## Package interface

Re-exports `State`, `StowNode`, `spec`, `initial_state`, and `step`.

## Interactions

MOVING emits the stow `StowReference` under the stow envelope; HELD emits
inhibit. Completion requires `stow_complete` plus fresh confirmed feedback.
The timeout latches `timeout_latched` so the fault and SAFE request emit
exactly once.

## Constraints

Pure; no `SystemMode`; no imaging or inference; ordinary target tolerance
alone does not confirm arrival.

## Related documents

- [`flight.payload.graphs`](../graphs.md)
- [`flight.payload.graphs.base`](base.md)
