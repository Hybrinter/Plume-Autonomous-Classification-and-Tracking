# tools.ml_models.export.accept

**Source:** `packages/tools/src/tools/ml_models/export/accept.py`
**Kind:** module
**Status:** implemented

## Purpose

This module is the acceptance gate boundary for an exported two-input
ONNX artifact. Artifact integrity and preprocessing checks precede
exhaustive shared test evaluation, then legacy acceptance thresholds are
applied to the measured quality metric and batch-one CPU latency.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `accept_artifact` | function | Runs the validation gate and returns a `Result` report |

## Inputs and outputs

`accept_artifact(artifact, manifest, dataset, *, min_iou=0.5,
min_accuracy=0.9, max_latency_ms=20.0) -> Result[dict[str, object], str]`.

The returned dict carries `sha256`, `hash_ok`, `contract_ok`,
`quality_ok`, `latency_ok`, `accepted`, the gated `metric` name, the
`quality_policy` label, `threshold`, worst observed latency, and the full
`evaluation` evidence. The function does not write a report; the CLI owns
publication of `.acceptance.json`.

## Behavior

1. Rejects non-finite, boolean, or out-of-range quality and latency
   thresholds.
2. Requires exactly one finished dataset directory and loads its
   `dataset.json`.
3. Requires bands, unit norm, and `gsd_reference_m` to match the model
   manifest, and each test shard's spatial size to match a declared
   fixed `input_shape` height/width; a dataset with no eligible test
   rows is refused.
4. Opens the artifact through the hash-first session boundary inside a
   `torch.nn.Module` adapter that records batch-one CPU latency per call
   and requires a single float32 logit output.
5. Runs `evaluate_split` on the `test` split with `batch_size=1` on
   `cpu`. Classifier acceptance reads the `accuracy` metric; segmentor
   acceptance reads `foreground_iou_mean_all_annotated_images`.
6. Applies the legacy thresholds: classifier `accuracy >= min_accuracy`,
   segmentor all-annotated-image IoU `>= min_iou`, and worst batch-one
   latency `<= max_latency_ms`. `accepted` requires both gates.

The segmentor gate deliberately uses the legacy all-annotated-image IoU:
verified-empty images count toward acceptance, matching the previous
acceptance convention. This differs from the positive-truth-image IoU
headline (`foreground_iou_mean_positive_images`) used for checkpoint
selection, which excludes empty masks. Both are explicit policies, not
substitutes for each other.

## Errors and faults

Returns `Err` for invalid thresholds, a missing or incompatible dataset,
manifest load failure, input-shape mismatch, session/artifact validation
failure, an evaluation error, an unavailable or unsupported quality
metric, or missing/misaligned latency evidence.

## Messages

None.

## Configuration

Threshold arguments only; `max_latency_ms` defaults to the flight
`inference_timeout_ms` value of 20.0. The ONNX runtime is imported lazily
through `export.session`.

## Constraints

- The supplied dataset must meet its per-source threshold.
- Latency figures are batch-one CPU timings on this host, not the
  on-board budget or target-hardware qualification.
- Evaluation and availability errors return `Err` without a report;
  measured quality or latency rejection returns a record with
  `accepted` false, which the CLI may persist.

## Related documents

- [tools.ml_models.export](../export.md)
- [tools.ml_models.export.session](session.md)
- [tools.ml_models.analysis.evaluate](../analysis/evaluate.md)
