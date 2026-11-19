# tools.ml_models.analysis.metrics.boundary

**Source:** `packages/tools/src/tools/ml_models/analysis/metrics/boundary.py`
**Kind:** module

## Purpose

Explicit interior mask boundaries, inclusive tolerance hits, and
symmetric distances. Four-neighbour erosion treats outside-image pixels
as background. Distances pool both directed nearest-boundary pixel
distances; HD95 is their linear 95th percentile. Empty-boundary
distances are unavailable, not fabricated.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `BoundaryRow` | dataclass | One explicit mask pair; hits use the stated tolerance, distances need both boundaries; records the resolved raw-mask threshold as provenance |
| `BoundaryEvidence` | dataclass | Pooled boundary hits plus equal-image conditional distance summaries |
| `binary_masks` | function | Validated aligned binary mask pair; `Result` boundary |
| `validate_gsd` | function | Optional finite positive lateral/along GSD; `Result` boundary |
| `score_boundary` | function | Per-image row computation; `Result` boundary |
| `aggregate_boundary` | function | Cohort pooling over rows; `Result` boundary |

## Inputs and outputs

`binary_masks(predicted, truth) ->
Result[tuple[ndarray, ndarray], str]` validates one aligned binary
`H`-by-`W` or singleton-channel mask pair. `validate_gsd(gsd) ->
Result[tuple[float, float] | None, str]` keeps unknown ground geometry
unavailable and rejects invalid axes or area.

`score_boundary(predicted, truth, *, gsd=None, cfg=None) ->
Result[BoundaryRow, str]` takes explicit binary masks, an optional GSD
pair, and an optional `ScoreConfig`.

`aggregate_boundary(rows) -> Result[BoundaryEvidence, str]` takes a
nonempty tuple of rows sharing one tolerance and raw-mask threshold
convention.

## Behavior

Interior boundaries are the foreground pixels removed by four-neighbour
erosion, with outside-image pixels treated as background. Each boundary
pixel's nearest distance to the other mask's boundary is pooled across
both directions. Hits count pooled pixels within the inclusive
configured tolerance — `boundary_tolerance_px`, or
`boundary_tolerance_m` when a physical tolerance is configured, which
requires recorded GSD and overrides the pixel tolerance. ASD is the
pooled-distance mean and HD95 its linear 95th percentile, each in
pixels and, when GSD is recorded, in metres with anisotropic sampling.
When either boundary is empty, hits are zero and distances are
unavailable with a reason.

Aggregation reports pooled boundary precision, recall, and harmonic F1
over boundary pixels, exact missed/spurious/empty image counts, and
equal-image means of the conditional ASD and HD95 distances over images
where both boundaries are nonempty and the requested geometry exists.

## Errors and faults

Misaligned, empty, or non-binary masks, invalid GSD, a physical
tolerance without recorded GSD, non-finite ground distances, an empty
row list, rows with non-finite or out-of-range recorded thresholds, and
rows scored under different tolerance or raw-mask threshold conventions
all return `Err`.

## Messages

None.

## Configuration

`ScoreConfig` supplies `boundary_tolerance_px`, `boundary_tolerance_m`,
and `mask_probability_threshold`. Rows carry their resolved tolerance,
unit, and raw-mask threshold as supplied provenance; the threshold names
the configured raw-score cut that produced the binary inputs and is
never inferred from mask contents. Mixed conventions cannot merge
silently.

## Constraints

- Distances are conditional on both boundaries being nonempty; the
  missed/spurious/empty image counts carry detection coverage.
- Metre distances use anisotropic local GSD: x lateral, y along.
- Inclusive tolerance is a diagnostic setting, not a mission acceptance
  criterion.

## Related documents

- [`tools.ml_models.analysis.metrics`](../metrics.md)
- [`tools.ml_models.analysis.metrics.spatial`](spatial.md)
- [`tools.ml_models.analysis.metrics.definitions`](definitions.md)
