# tools.ml_models.analysis.plots

**Source:** `packages/tools/src/tools/ml_models/analysis/plots/`
**Kind:** package

## Purpose

The plots package renders figures over frozen evidence. `common`
exports rendered figures to bundle bytes and dispatches the
render-only bundle boundary; `dataset`, `training`, and `model` render
frozen recipes. This package replaces the removed report-figure
module; the legacy `history_figures`, `overlay_figures`,
`failure_figures`, and `save_figures` APIs are gone.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`common`](plots/common.md) | module | Figure export to bundle bytes and the render-only bundle boundary |
| [`dataset`](plots/dataset.md) | module | Frozen dataset figure rendering |
| [`training`](plots/training.md) | module | Frozen training-history figure rendering |
| [`model`](plots/model.md) | module | Shared frozen model-chart recipe rendering |
| [`classifier`](plots/classifier.md) | module | Classifier family binding over the model renderer |
| [`segmentation`](plots/segmentation.md) | module | Segmentor family binding over the model renderer |
| [`generalization`](plots/generalization.md) | module | Generalization family binding over the model renderer |

## Package interface

`tools.ml_models.analysis.plots.__init__` carries a module docstring only.
The render boundary lives in `tools.ml_models.analysis.plots.common`.

## Interactions

`render_analysis` verifies a published bundle by checksum, then
dispatches on the summary kind: dataset bundles render through
`dataset_render.render_dataset_bundle` and model bundles through
`model_render.render_model_bundle`. Plotting consumes frozen evidence
only and never reruns inference.

## Constraints

- Rendering consumes frozen recipes only and never reruns inference.
- Figures derive only from captured evidence records.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.cli`](../cli.md)
