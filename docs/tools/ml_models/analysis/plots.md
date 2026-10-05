# tools.ml_models.analysis.plots

**Source:** `packages/tools/src/tools/ml_models/analysis/plots/`
**Kind:** package
**Status:** stub

## Purpose

The plots package is the scaffold for figure rendering over frozen
captured evidence; no figure is produced yet. This package replaces the
removed report-figure module; the legacy `history_figures`,
`overlay_figures`, `failure_figures`, and `save_figures` APIs are gone.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`common`](plots/common.md) | module | Figure/export conventions and the render boundary (unavailable) |
| [`dataset`](plots/dataset.md) | module | Dataset-analysis figures (scaffold) |
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
