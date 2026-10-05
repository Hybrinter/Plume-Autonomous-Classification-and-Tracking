# flight.payload.graphs.init.ready

**Source:** `packages/flight/src/flight/payload/graphs/init/ready.py`
**Kind:** pure module

## Purpose

The READY node emits one `VERIFY_INIT` intent, then waits for an
`InitVerificationResult` on the current activation. A VERIFIED result with
nonempty evidence emits exactly one `INIT_COMPLETE` request while the graph
stays READY for external selection. A FAILED verification latches
`failed`, raises `GIMBAL_FAULT`, and requests SAFE once.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `step` | function | Inhibit outcome; intents, request, or latched failure |

## Inputs and outputs

`step(state, inputs, params) -> (State, NodeOutcome[InitNode])`.

## Behavior

`step` emits `VERIFY_INIT` once, then honors `inputs.verification` only
on ticks after the intent was issued. A VERIFIED result with nonempty
evidence emits one `INIT_COMPLETE` request and stays READY; FAILED latches
`failed` with `GIMBAL_FAULT` and one SAFE request.

## Errors and faults

`GIMBAL_FAULT` on verification failure; SAFE intent once on the latched
failure.

## Messages

None.

## Configuration

Reads `PactConfig` through `GraphParameters`.

## Constraints

Pure; no default pass. Missing or PENDING verification waits, and VERIFIED
never creates a local IDLE.

## Related documents

- [`flight.payload.graphs.init`](../init.md)
