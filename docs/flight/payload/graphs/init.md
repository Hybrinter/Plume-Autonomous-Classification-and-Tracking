# flight.payload.graphs.init

**Source:** `packages/flight/src/flight/payload/graphs/init`
**Kind:** package

## Purpose

The INIT graph walks `SELFTEST -> MODEL_LOAD -> HOME -> READY` on
activation-scoped effect completions, then waits for explicit verification
before requesting IDLE. Stale, unsolicited, pending, or duplicate results
never advance progress; failures latch once with a fault and one SAFE
request.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`state`](init/state.md) | pure module | `InitNode` and INIT `State` records |
| [`selftest`](init/selftest.md) | pure module | SELFTEST node step |
| [`model_load`](init/model_load.md) | pure module | MODEL_LOAD node step |
| [`home`](init/home.md) | pure module | HOME node step |
| [`ready`](init/ready.md) | pure module | READY node step and verification wait |
| [`graph`](init/graph.md) | pure module | Spec, edges, initial state, and step |

## Package interface

Re-exports `State`, `InitNode`, `spec`, `initial_state`, and `step`.

## Interactions

Each non-motion effect node inhibits and issues one `EffectIntent`; HOME
poses to the configured home target with fresh feedback; READY emits one
`VERIFY_INIT` intent and one `INIT_COMPLETE` request after a VERIFIED result
with nonempty evidence, then stays READY for external selection.

## Constraints

Pure; no `SystemMode`; effect intents issue once per activation and only
already-issued results may advance the chain.

## Related documents

- [`flight.payload.graphs`](../graphs.md)
- [`flight.payload.graphs.base`](base.md)
