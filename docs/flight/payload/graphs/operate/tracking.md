# flight.payload.graphs.operate.tracking

**Source:** `packages/flight/src/flight/payload/graphs/operate/tracking.py`
**Kind:** pure module

## Purpose

The TRACKING node reproduces the control outer-loop composition: fresh CoG
intersect on an accepted sample, acquire residual reset at the shutter,
encoder/nominal/reference/vision events into the residual history, estimate
replay at the encoder time, and the scene-plus-residual rate law.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `step` | function | Updated state plus a `RateReference` or inhibit outcome |

## Inputs and outputs

`step(state, inputs, params) -> (State, NodeOutcome[OperateNode])`.

## Behavior

1. Stale or missing feedback inhibits.
2. An accepted sample with blobs clears the miss counter, refreshes liveness,
   and may cold-start the residual under the acquire policy.
3. The scene rate is nominal plus residual; only the `Kp * e` relative
   term is clipped to the smear cap, and `rate_decision` bounds the
   requested rate under the science envelope.
4. A supplied `TickInputs.reference_change` takes precedence over the
   derived CoG rebase in the residual history. The computed `RateDecision`,
   current azimuth residual, and vision disposition are retained on the
   state for the shell to publish.

## Errors and faults

None directly; flagged samples are handled by the graph step.

## Messages

None.

## Configuration

Reads `PactConfig` through `GraphParameters`.

## Constraints

Pure; unknown navigation keeps the scene nominal at zero as a computational
fallback only.

## Related documents

- [`flight.payload.graphs.operate`](../operate.md)
