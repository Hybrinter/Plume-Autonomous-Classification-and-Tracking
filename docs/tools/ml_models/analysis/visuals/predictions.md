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
Classifier families only; segmentation returns `Err` until PR14.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `PredictionPreview` | dataclass | Bound preview record: row, path, checksum, size, display mapping |
| `PredictionPreviewCapture` | dataclass | Immutable preview files paired with frozen galleries |
| `render_prediction_visuals` | function | Export gallery pages into bundle bytes |

## Inputs and outputs

`render_prediction_visuals(captured, cfg) ->
Result[RenderedDatasetFigures, str]` emits pages of at most four
panels under `visuals/predictions/<gallery>_<page>.<fmt>` and one
output record per gallery reusing the
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

## Errors and faults

Checksum/size mismatches, missing image/target/logits entries,
non-decodable or truncated archives, non-canonical image
dtype/shape/domain, target or raw-logit disagreement with the exact
captured Python floats, out-of-bounds display indices, duplicate
preview keys or file paths, duplicate gallery identifiers or output
names, unsafe gallery or preview tokens, a chosen key bound to a
preview row carrying different scalars, and non-classifier galleries
return `Err` before anything is published.

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
