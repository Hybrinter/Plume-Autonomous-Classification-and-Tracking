# tools.ml_models.export.accept

**Source:** `packages/tools/src/tools/ml_models/export/accept.py`
**Kind:** module

## Purpose

This module is the acceptance gate boundary for an exported artifact. The
artifact/dataset contract validations remain live; scoring is unavailable
until the evidence evaluation phase lands.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `accept_artifact` | function | Runs the validation gate and returns a `Result` report |

## Behavior

1. Rejects non-finite or out-of-range quality and latency thresholds.
2. Requires exactly one finished dataset directory and loads its
   `dataset.json`.
3. Requires bands, unit norm, `gsd_reference_m`, and each test shard's
   spatial size (when the model declares fixed input H/W) to match the
   model manifest.
4. Returns `Err` with an explicit unavailable message once every
   validation passes. No session opens, no inference runs, and no
   acceptance report is written.

## Errors and faults

Returns `Err` for invalid thresholds, an empty or incompatible dataset,
a manifest load failure, or an input-shape mismatch. Valid invocations
still return `Err` while scoring is unavailable.

## Inputs and outputs

`accept_artifact(artifact, manifest, dataset, min_iou=0.5,
min_accuracy=0.9, max_latency_ms=20.0) -> Result[dict[str, object], str]`.

## Messages

None.

## Configuration

Threshold arguments only; `max_latency_ms` defaults to the flight `inference_timeout_ms` value of 20.0.

## Constraints

- The supplied dataset must meet its per-source threshold.
- Latency figures are batch-one CPU timings, not the on-board budget.
- A failed or unavailable gate never writes an acceptance report.

## Related documents

- [tools.ml_models.export](../export.md)
- [tools.ml_models.export.session](session.md)
