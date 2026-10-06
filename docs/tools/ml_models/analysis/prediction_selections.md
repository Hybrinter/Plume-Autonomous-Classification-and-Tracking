# tools.ml_models.analysis.prediction_selections

**Source:** `packages/tools/src/tools/ml_models/analysis/prediction_selections.py`
**Kind:** module

## Purpose

Whole-cohort deterministic prediction gallery selection over frozen
scalar rows. Selections use the same seed/key priority and failure
ordering as bounded capture. High-confidence classifier errors are
globally selected from all errors by unweighted BCE, monotone in
wrong predicted-class confidence. Missing cache entries remain
missing; chosen failures are never replaced with easier examples.
No image or model is read.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `PredictionGallery` | dataclass | Selected identities, availability, and method |
| `prediction_key` | function | Content key used by bounded prediction capture |
| `prediction_priority` | function | Exact capture seed/key tie priority |
| `prediction_gallery_data` | function | Freeze representative and global error families |

## Inputs and outputs

`prediction_gallery_data(evidence, rows, cfg) ->
Result[tuple[PredictionGallery, ...], str]` takes frozen
`SplitEvidence`, the complete captured-row cohort, and
`CaptureConfig`.

## Behavior

Families are `representative` (seeded whole-cohort display selection;
not independent/random sampling), `false_positive`, `false_negative`,
plus `high_confidence_error` for classifiers (descending captured
unweighted BCE) or `worst` for other tasks (descending captured
failure score). Each gallery records an `AvailabilityRecord` named
`prediction_visual:<split>:<family>`; disabled budgets keep `SKIPPED`
and empty families keep `UNAVAILABLE` with explicit reasons.
Selection honors `examples_per_family` as an upper bound only.

## Errors and faults

Returns `Err` when capture coverage is incomplete or
identity-mismatched (via `validate_figure_rows`), or when classifier
failure ordering does not match the frozen unweighted BCE.

## Messages

None.

## Configuration

`CaptureConfig.seed`, `max_preview_images`, and `examples_per_family`
bound the selection; they do not tune scores.

## Constraints

- Ordering is whole-cohort and deterministic; input traversal order
  does not change selections.
- Chosen rows are never swapped for easier cached examples.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.visuals.predictions`](visuals/predictions.md)
- [`tools.ml_models.analysis.capture`](capture.md)
