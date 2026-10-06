# tools.ml_models.analysis.plots

**Source:** `packages/tools/src/tools/ml_models/analysis/plots/`
**Kind:** package

## Purpose

The plots package renders figures over frozen evidence. `common`
exports rendered figures to bundle bytes and keeps the general render
boundary unavailable; `dataset` renders frozen dataset figure recipes.
This package replaces the removed report-figure module; the legacy
`history_figures`, `overlay_figures`, `failure_figures`, and
`save_figures` APIs are gone.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`common`](plots/common.md) | module | Figure export to bundle bytes and the unavailable render boundary |
| [`dataset`](plots/dataset.md) | module | Frozen dataset figure rendering |
| [`training`](plots/training.md) | module | Training-history figures (scaffold) |
| [`classifier`](plots/classifier.md) | module | Classifier-evidence figures (scaffold) |
| [`segmentation`](plots/segmentation.md) | module | Segmentation-evidence figures (scaffold) |
| [`generalization`](plots/generalization.md) | module | Generalization-evidence figures (scaffold) |

## Package interface

`tools.ml_models.analysis.plots.__init__` carries a module docstring only.
The render boundary lives in `tools.ml_models.analysis.plots.common`.

## Interactions

The `render_analysis` boundary is unavailable; plotting consumes frozen
evidence only and never reruns inference.

## Constraints

- Rendering is unavailable until the plotting phase lands.
- Figures derive only from captured evidence records.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.cli`](../cli.md)
