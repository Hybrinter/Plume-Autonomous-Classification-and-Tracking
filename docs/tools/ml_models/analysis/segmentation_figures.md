# tools.ml_models.analysis.segmentation_figures

**Source:** `packages/tools/src/tools/ml_models/analysis/segmentation_figures.py`
**Kind:** module

## Purpose

Extent and localization chart recipes from frozen explicit-mask
evidence. Positive-mask distributions stay separate from verified
negatives and positive-label empty annotations. Matched centroid
ECDFs are conditional; copied localization-success curves retain
misses in their denominator. No component matching, boundary
measurement, or dense inference is rerun.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `segmentation_figure_data` | function | Freeze the complete segmentor figure inventory |

## Inputs and outputs

`segmentation_figure_data(evidence, rows) ->
Result[tuple[ModelFigure, ...], str]` requires segmentor
`SplitEvidence` and the complete captured-row cohort.

## Behavior

Emitted figures include scalar/extent distributions
(`foreground_iou`, `foreground_dice`, image losses, truth and
predicted areas in px and m2, retained blob counts) split into
positive-truth, verified-negative-empty, and positive-label-empty
cohorts; per-image foreground precision/recall distributions with
eligible denominators; extent-error distributions; conditional
boundary ASD/HD95 distributions in px and m; captured
`threshold_positive_image_*` curves, miss-inclusive
`localization_success_*` curves, and approximate
`pixel_*_histogram`/calibration curves with their recorded
operating points; one `metric_*` figure per captured metric; and an
explicit `spatial_capture` availability figure when no
component/boundary evidence exists. Captured curves keep raw x/null
coordinates; all-null curves carry the recorded unavailability
reason.

## Errors and faults

Returns `Err` for non-segmentor evidence, incomplete explicit-mask
scalar capture, non-integer or inconsistent pixel counts, or
captured areas disagreeing with the frozen pixel counts.

## Messages

None.

## Configuration

None.

## Constraints

- Pixel counts must satisfy `tp + fp + tn + fn == shard pixels` and
  agree with the captured areas exactly; recorded `foreground_iou`
  and `foreground_dice` must match the same counts within 1e-12
  (both-empty overlap is defined as 1).
- Conditional matched-centroid statistics never include misses; the
  success curves copy frozen coordinates that do.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.model_figures`](model_figures.md)
- [`tools.ml_models.analysis.plots.segmentation`](plots/segmentation.md)
