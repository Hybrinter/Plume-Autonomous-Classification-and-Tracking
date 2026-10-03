# tools.ml_models.export.accept

**Source:** `packages/tools/src/tools/ml_models/export/accept.py`
**Kind:** module

## Purpose

This module is the acceptance gate for an exported artifact: it evaluates the
validated two-input ONNX session over every test row of the supplied
finished dataset and gates quality plus worst batch-one latency.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `accept_artifact` | function | Runs the gate and returns a report dict |

## Behavior

1. Loads `dataset.json` from the dataset directory and requires bands,
   unit norm, `gsd_reference_m`, and each test shard's spatial size
   (when the model declares fixed input H/W) to match the model manifest.
2. Opens the artifact through `open_session` (hash and contract enforced).
3. Evaluates batch-one over the `test` split via `tools.ml_models.train.evaluate`;
   classifiers use `accuracy` and segmentors `mean_iou`.
4. Measures worst single-row session latency. These are batch-one CPU timings,
   not the 64-tile on-board budget.
5. Returns a report with `hash_ok`, `contract_ok`, `quality_ok`,
   `latency_ok`, `accepted`, `metric`, `threshold`,
   `worst_batch_one_latency_ms`, `max_latency_ms`, and the evaluation payload.
   The CLI writes it to `artifact.with_suffix(".acceptance.json")` tied to
   the artifact SHA-256, dataset hashes, and thresholds.

## Errors and faults

Raises `ValueError`/`OSError` for invalid thresholds, an incompatible
dataset, or session failures; the `accepted` flag carries gate results.

## Inputs and outputs

`accept_artifact(artifact, manifest, dataset, min_iou=0.5, min_accuracy=0.9, max_latency_ms=20.0)` returns a report dict.

## Messages

None.

## Configuration

Threshold arguments only; `max_latency_ms` defaults to the flight `inference_timeout_ms` value of 20.0.

## Constraints

- The supplied dataset must meet its per-source threshold.
- Latency figures are batch-one CPU timings, not the on-board budget.

## Related documents

- [tools.ml_models.export](../export.md)
- [tools.ml_models.export.session](session.md)
- [tools.ml_models.train.evaluate](../train/evaluate.md)
