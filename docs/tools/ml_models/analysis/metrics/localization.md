# tools.ml_models.analysis.metrics.localization

**Source:** `packages/tools/src/tools/ml_models/analysis/metrics/localization.py`
**Kind:** module

## Purpose

Unfiltered truth components, retained prediction components, and
cardinality-first matching. Four-connected raster-order components are
mask geometry, not independent physical plumes. Eligible IoU matching
maximizes match count before total IoU. Conditional centroid errors are
separate from miss-inclusive coverage.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `MaskComponent` | dataclass | Raster-order four-connected component with pixel-center geometry |
| `ComponentMatch` | dataclass | One eligible one-to-one pair; displacements are prediction minus truth |
| `LocalizationRow` | dataclass | Compact component/pair evidence; no dense predictions are retained |
| `LocalizationEvidence` | dataclass | Component counts, conditional errors, and miss-inclusive success curves |
| `match_components` | function | Cardinality-first eligible IoU assignment; `Result` boundary |
| `score_localization` | function | Per-image row computation; `Result` boundary |
| `aggregate_localization` | function | Cohort pooling over rows; `Result` boundary |

## Inputs and outputs

`match_components(ious, minimum) -> Result[tuple[tuple[int, int], ...],
str]` takes a finite numeric two-dimensional IoU matrix with entries and
`minimum` in `[0, 1]` and returns eligible index pairs.

`score_localization(logits, truth, *, gsd=None, cfg=None) ->
Result[LocalizationRow, str]` takes aligned `H`-by-`W` or
singleton-channel `(1, H, W)` arrays, an optional `(lateral,
along-track)` GSD pair in metres, and an optional `ScoreConfig`.

`aggregate_localization(rows) -> Result[LocalizationEvidence, str]`
takes a nonempty tuple of rows scored under identical component and
matching conventions.

## Behavior

Truth components are labelled four-connected in raster order with no
area filter; the prediction is thresholded at
`blob_probability_threshold` and its components are retained only at
`min_blob_area_px`. Pairwise component IoUs come from exact label
intersection counts. `match_components` maximizes eligible match
cardinality first, then summed IoU, using zero-cost dummy matches.
Each matched pair records prediction-minus-truth centroid displacement
in pixels and, when GSD is recorded, in metres as
`hypot(dx * lateral_GSD, dy * along_GSD)`. Split truth counts record
truths overlapping multiple retained predictions; merge counts record
retained predictions overlapping multiple truths.

Aggregation pools component counts into exact matched/unmatched truth
and prediction metrics, `component_precision`, `component_recall`, and
`component_f1`, split/merge counts, and conditional matched centroid
error means, medians, and linear p95 in pixels and metres.
`localization_success_px` and `localization_success_m` curves evaluate
the truth-component success fraction at every observed matched distance
threshold; the denominator includes every truth component carrying the
requested geometry, including unmatched misses, and metre coverage
reports truth components missing recorded GSD explicitly.

## Errors and faults

Misaligned or empty masks, non-binary values, a non-finite or
out-of-range IoU matrix or minimum, invalid GSD, non-finite matched
ground distances, an empty row list, and rows scored under different
thresholds or matching conventions all return `Err`.

## Messages

None.

## Configuration

`ScoreConfig` supplies `blob_probability_threshold`,
`min_blob_area_px`, and `match_iou_min`. Rows carry their resolved
settings so mixed conventions cannot merge silently.

## Constraints

- Truth components are not area-filtered and do not prove separate
  physical plumes.
- Prediction thresholding uses finite logits, not float32 flight
  probability rounding.
- Centroid distributions are conditional on eligible matching; success
  curves include misses.
- Ground distances and areas are local-GSD approximations.

## Related documents

- [`tools.ml_models.analysis.metrics`](../metrics.md)
- [`tools.ml_models.analysis.metrics.spatial`](spatial.md)
- [`tools.ml_models.analysis.metrics.definitions`](definitions.md)
- [`tools.ml_models.analysis.metrics.inputs`](inputs.md)
