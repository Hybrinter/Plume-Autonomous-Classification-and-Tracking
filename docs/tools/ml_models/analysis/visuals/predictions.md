# tools.ml_models.analysis.visuals.predictions

**Source:** `packages/tools/src/tools/ml_models/analysis/visuals/predictions.py`
**Kind:** module

## Purpose

Prediction-overlay visuals over captured evaluation evidence. Renders
the fixed `PredictionGallery` selections against immutable preview
bytes supplied by capture. Every chosen panel verifies the NPZ
checksum and size before `np.load` on an in-memory buffer
(`allow_pickle=False`), then requires the exact canonical float32
unit image, the recorded binary target, and the exact Python float
of the captured float32 raw logit — no
coercion, resize, normalization, substitution, or re-selection.
Segmentor rows take shape-(1,H,W) float32 masks and logits through
the frozen `segmentation_display_data` helper, whose recomputed
pixel counts must agree exactly with captured metrics. Galleries
mixing classifier and segmentor cohorts are refused.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `PredictionPreview` | dataclass | Bound preview record: row, path, checksum, size, display mapping |
| `PredictionPreviewCapture` | dataclass | Immutable preview files paired with frozen galleries |
| `LoadedPrediction` | dataclass | Verified cache arrays plus the frozen segmentor display |
| `render_prediction_visuals` | function | Export gallery pages into bundle bytes |

## Inputs and outputs

`render_prediction_visuals(captured, cfg) ->
Result[RenderedDatasetFigures, str]` emits classifier pages of at
most four panels and segmentor pages of exactly one example under
`visuals/predictions/<gallery>_<page>.<fmt>`, plus one output record
per gallery reusing the
`prediction_visual:<split>:<family>` name.

## Behavior

Each panel shows the captured input through the recorded semantic
channel mapping and a title with tile id, GSD, truth, recorded
predicted class (read from truth plus error flags, never re-scored),
positive probability, and unweighted BCE. The family
`selection_method` labels every page prominently. Chosen rows whose
preview was omitted by the capture budget render an explicit omitted
panel and the gallery output becomes `UNAVAILABLE` with reason
"Chosen prediction previews were omitted by the capture budget";
supplied `SKIPPED`/`UNAVAILABLE` family states and reasons are
preserved, and chosen identities are never swapped.

Each segmentor page draws a fixed 2x3 extent grid: the captured
input through the semantic channel mapping, the explicit truth mask
and raw binary mask in fixed 0/1 grayscale, the captured-logit
probability on a labelled 0..1 colorbar, an FP/FN pixel map (FP red,
FN blue, true positives gray), and a frozen component-match overlay
drawing every recorded truth and retained prediction bounding box
and centroid, lines only between frozen matched pairs, and `x`
markers on unmatched components. The footer copies the raw-mask
threshold, blob threshold/minimum area, matching IoU, match and
unmatched counts, GSD, and notes that centroid distances are
conditional on matched pairs while misses carry no error. Missing
or dangling component IDs and nonfinite or off-image geometry fail
closed before plotting.

## Errors and faults

Checksum/size mismatches, missing image/target/logits entries,
non-decodable or truncated archives, non-canonical image
dtype/shape/domain, target or raw-logit disagreement with the exact
captured Python floats, out-of-bounds display indices, duplicate
preview keys or file paths, duplicate gallery identifiers or output
names, unsafe gallery or preview tokens, a chosen key bound to a
preview row carrying different scalars, mixed-task galleries, and
any frozen-evidence disagreement reported by
`segmentation_display_data` return `Err` before anything is
published.

## Messages

None.

## Configuration

`PlotConfig` controls dimensions, DPI, font size, and formats.

## Constraints

- Selections are consumed verbatim; the renderer never reselects rows
  or reads sources, captures, or models.
- Preview bytes are hash- and size-verified before decoding.

## Related documents

- [`tools.ml_models.analysis.visuals`](../visuals.md)
- [`tools.ml_models.analysis.prediction_selections`](../prediction_selections.md)
- [`tools.ml_models.analysis.prediction_artifacts`](../prediction_artifacts.md)
