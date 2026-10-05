# tools.ml_models.analysis.metrics.segmentation

**Source:** `packages/tools/src/tools/ml_models/analysis/metrics/segmentation.py`
**Kind:** module

## Purpose

Exact per-image foreground evidence and bounded pixel diagnostics.
Positive-truth overlap excludes empty masks, and explicit verified empty
masks form their own false-mask population. Pixel ranking uses a labeled
histogram approximation; overlap, areas, counts, probability losses, and
bin sums are exact.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `PixelHistogram` | dataclass | Fixed-memory class counts and probability sums on explicit bin grids |
| `ThresholdCounts` | dataclass | Exact `tp`/`fp`/`tn`/`fn` at one probability threshold |
| `SegmentationRow` | dataclass | Scalar sufficient statistics for one annotated image |
| `SegmentationEvidence` | dataclass | Aggregate metrics, curves, support, merged histogram, output availability |
| `score_segmentation_image` | function | Per-image row computation; `Result` boundary |
| `aggregate_segmentation` | function | Cohort aggregation over rows; `Result` boundary |

## Inputs and outputs

`score_segmentation_image` takes logit and mask array-likes in matching
`H`-by-`W` or singleton-channel `(1, H, W)` shape, a binary `label`, an
explicit `verified_empty` flag, an optional `(lateral, along-track)` GSD
pair in metres, and an optional `ScoreConfig`. It returns
`Result[SegmentationRow, str]`. `aggregate_segmentation` takes a
nonempty tuple of rows sharing one `ScoreConfig` and returns
`Result[SegmentationEvidence, str]`.

## Behavior

Per image, predictions use the log-odds decision at
`mask_probability_threshold` (threshold zero predicts all, one predicts
none, matching the classifier convention). Exact counts produce IoU
`TP/(TP+FP+FN)` and Dice `2*TP/(2*TP+FP+FN)`, each defined as one when
both masks are empty, plus unweighted per-pixel BCE and Brier means and
per-class Brier and probability residuals (null when the truth class is
absent). `n_predicted_blobs` counts four-connected components at
`blob_probability_threshold` that reach `min_blob_area_px`. A fixed-width
`PixelHistogram` records class counts and probability sums on
`pixel_histogram_bins` rank bins and `n_calibration_bins` calibration
bins, and every configured probability threshold gets exact confusion
counts. When GSD is supplied, pixel areas scale to square metres.

Aggregation distinguishes image- and pixel-weighted values: equal-image
IoU/Dice means are reported separately over nonempty-truth images and
all annotated images, so verified negatives cannot inflate the positive
headline; pooled global IoU/Dice/precision/recall use pixel counts; BCE
and Brier appear as equal-image means and as pixel-weighted means; area
errors are in pixels and, for GSD-annotated images, in square metres via
the local-GSD approximation; verified-empty cohorts get any-foreground,
any-blob, and mean-false-blob rates; foreground/background Brier and
residual means are weighted by truth-class pixel counts. Merged pixel
histograms yield approximate `pixel_roc_auc_histogram` and
`pixel_average_precision_histogram` (equal-width bins treated as ties,
`HISTOGRAM` curves) plus the exact occupied-bin
`pixel_expected_calibration_error`; pixel reliability and support curves
carry the
correlation caveat. `threshold_positive_image_iou` and
`threshold_positive_image_dice` curves sweep the configured threshold
grid over positive-truth images.

## Errors and faults

Invalid masks, non-binary labels, shape mismatches, non-finite inputs,
`verified_empty` on a positive label or nonzero mask, a negative label
with positive mask pixels, non-positive or non-finite GSD, an empty row
list, and mixed `ScoreConfig` rows all return `Err` before scoring.

## Messages

None.

## Configuration

`ScoreConfig` supplies `mask_probability_threshold`,
`blob_probability_threshold`, `min_blob_area_px`,
`pixel_histogram_bins`, `n_calibration_bins`, and
`probability_thresholds`. Rows carry their resolved config so mixed
settings cannot merge silently.

## Constraints

- Pixel ranking diagnostics are histogram approximations over
  fixed-width bins; overlap, counts, losses, and calibration bin sums
  are exact.
- Correlated pixels are not independent observations; pixel curves are
  diagnostics, not per-pixel inference.
- GSD areas are a local per-image approximation.
- Annotated rows require explicit truth. Contradictory labels and masks
  are refused. Empty masks without verified-negative status remain
  outside the verified-negative false-alarm population.

## Related documents

- [`tools.ml_models.analysis.metrics`](../metrics.md)
- [`tools.ml_models.analysis.metrics.inputs`](inputs.md)
- [`tools.ml_models.analysis.metrics.definitions`](definitions.md)
