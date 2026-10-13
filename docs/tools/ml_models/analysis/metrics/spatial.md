# tools.ml_models.analysis.metrics.spatial

**Source:** `packages/tools/src/tools/ml_models/analysis/metrics/spatial.py`
**Kind:** module

## Purpose

Shared per-image localization/boundary measurement and frozen aggregate
reduction. Blob localization uses its own blob threshold/area gate.
Boundary extent uses the raw configured mask threshold without that area
filter. This separation keeps a tiny true component visible as a miss
without erasing its truth.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SpatialRow` | dataclass | Compact component matching and boundary evidence for one annotated image |
| `SpatialEvidence` | dataclass | Whole-cohort metrics and miss-inclusive exact success curves |
| `score_spatial` | function | Per-image row computation; `Result` boundary |
| `aggregate_spatial` | function | Frozen aggregate reduction; `Result` boundary |

## Inputs and outputs

`score_spatial(logits, truth, *, gsd=None, cfg=None) ->
Result[SpatialRow, str]` takes aligned logit and mask array-likes, an
optional `(lateral, along-track)` GSD pair in metres, and an optional
`ScoreConfig`.

`aggregate_spatial(rows) -> Result[SpatialEvidence, str]` takes a tuple
of frozen rows and returns the combined localization and boundary
metrics plus the localization success curves.

## Behavior

Per image, `score_spatial` delegates component matching to
`score_localization` at `blob_probability_threshold` and
`min_blob_area_px`, then thresholds the same logits at the raw
`mask_probability_threshold` for `score_boundary`. The row is a compact
record of component geometry, matches, unmatched identifiers, boundary
counts and distances — no dense arrays are retained.

`aggregate_spatial` reduces the frozen rows through
`aggregate_localization` and `aggregate_boundary` and concatenates their
metrics; the curves come from the localization evidence. No inference,
re-scoring, or new reduction happens here.

## Errors and faults

Underlying scorer errors — mask, geometry, GSD, or convention
mismatches — propagate unchanged as `Err`. Empty or convention-mixed
row tuples fail inside the delegated aggregates.

## Messages

None.

## Configuration

`ScoreConfig` supplies `mask_probability_threshold`,
`blob_probability_threshold`, `min_blob_area_px`, `match_iou_min`, and
the boundary tolerances; see
[`tools.ml_models.analysis.config`](../config.md).

## Constraints

- The evaluator calls this module; callers never re-derive component or
  boundary statistics from dense predictions.
- The blob area gate applies to localization only; truth components are
  never area-filtered.
- Success-curve denominators are the truth components carrying the
  requested geometry, including unmatched misses; the
  `truth_components_missing_geometry` support count records components
  excluded for missing GSD.

## Related documents

- [`tools.ml_models.analysis.metrics`](../metrics.md)
- [`tools.ml_models.analysis.metrics.localization`](localization.md)
- [`tools.ml_models.analysis.metrics.boundary`](boundary.md)
- [`tools.ml_models.analysis.metrics.definitions`](definitions.md)
- [`tools.ml_models.analysis.evaluate`](../evaluate.md)
- [`tools.ml_models.analysis.capture`](../capture.md)
