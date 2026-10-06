# tools.ml_models.analysis.evaluate

**Source:** `packages/tools/src/tools/ml_models/analysis/evaluate.py`
**Kind:** module
**Status:** implemented

## Purpose

This module runs one exhaustive split evaluation over every eligible row
of a verified dataset split with one conditioned model and returns typed
`SplitEvidence`. Scoring is delegated to the shared classifier and
segmentation evaluators; this module owns cohort traversal, input
validation, canonical augmentation views, optional objective records, and
capture integration.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `evaluate_split` | function | Exhaustive split evaluation boundary |
| `canonical_array` | function | Inverse dihedral view for channel-first arrays |

## Inputs and outputs

`evaluate_split(model, dataset, manifest, cfg, objective=None,
capture=None) -> Result[SplitEvidence, str]`.

`model` is called as `model(image, encoded_gsd)` and must return one
finite float32 logit tensor shaped like the targets. `objective` is an
optional `PlumeLoss` whose `per_sample_components` produce separate
`objective_*` records; it plays no part in scoring. `capture` is an
optional `CaptureSink` that receives every evaluated row.

## Behavior

1. Re-reads `dataset.json` and requires it to equal the supplied verified
   manifest, then selects shards matching `cfg.kind`/`cfg.split`.
2. Validates shard row counts, positive counts, binary labels, finite
   positive GSD, legal axis-preserving augmentation elements, and unique
   tile/bin/GSD identities. Validation and test splits require identity
   views.
3. Train rows pick one canonical view per augmentation identity: the
   `id` element when present, else the variant with the lowest
   `ELEMENT_NAMES` index, inverted once through `canonical_array` on the
   image and, for segmentors, the target. Excluded copies and inverted
   views are counted in warnings.
4. Runs `cfg.batch_size` batches on `cfg.device`; logit shape, float32
   dtype, and finiteness are required. The model is run under
   `model.eval()` and `torch.inference_mode()`, and every module's prior
   `training` flag is restored on exit.
5. Per row, classifier logits or a `SegmentationRow` plus a `SpatialRow`
   (from `score_spatial` at the configured blob, mask, and tolerance
   settings) plus optional `objective_*` components feed a total cohort
   and one `gsd_bin` stratum cohort per `bin_id`; classifier cohorts
   score through `score_classifier`, segmentor cohorts through the
   pooled-histogram `SegmentationAccumulator` plus `aggregate_spatial`
   over the frozen spatial rows, which appends component, boundary, and
   localization-success evidence unchanged.
6. When `capture` is supplied, each evaluated row is offered as a
   `CaptureRow` with evaluator-derived metrics, failure score, and
   false-positive/false-negative flags, plus canonical image/target/logit
   arrays (`array_view` `CANONICAL`). Segmentor rows carry their frozen
   `SpatialRow` in `CaptureRow.spatial`; classifier rows carry `None`.
7. The returned `SplitEvidence` carries cohort metrics, curves, support
   with an `n_groups` named count, `gsd_bin` strata, capture artifact
   references, and warnings for excluded copies, inverted views, and
   unavailable metrics.

## Errors and faults

Manifest disagreement, missing task/split rows, repeated spatial shards,
count mismatches, non-binary labels or targets, bad GSD or elements,
repeated augmentation identities, non-identity validation/test views,
misaligned or non-float32 or non-finite logits, scorer errors, and
capture errors all return `Err`. On failure the sink is aborted with the
error and closed; its close error is reported, and a partial capture
never verifies as complete. Model exceptions are converted to `Err` and
module modes are always restored.

## Messages

None.

## Configuration

`EvaluationConfig` supplies `kind`, `split`, `batch_size`, `device`, and
`score`; see [`tools.ml_models.analysis.config`](config.md). Model and
objective placement on the device is caller-owned.

## Constraints

- Evaluation consumes only verified finished datasets; identities and
  counts are re-checked against `dataset.json` bytes.
- Model outputs are required float32; non-conforming outputs fail.
- Canonical fallback exists for train rows only; validation and test
  require identity views.
- Strata are `gsd_bin` cohorts evaluated independently; no other grouping
  is invented.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.contracts`](contracts.md)
- [`tools.ml_models.analysis.capture`](capture.md)
- [`tools.ml_models.analysis.metrics.segmentation`](metrics/segmentation.md)
- [`tools.ml_models.analysis.metrics.spatial`](metrics/spatial.md)
- [`tools.ml_models.analysis.metrics.classifier`](metrics/classifier.md)
