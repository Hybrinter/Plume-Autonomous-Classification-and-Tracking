# tools.ml_models.analysis.visuals

**Source:** `packages/tools/src/tools/ml_models/analysis/visuals/`
**Kind:** package

## Purpose

The visuals package assembles visual evidence: deterministic bounded
gallery selection, dataset preview-gallery rendering, and
classifier prediction-overlay rendering over verified preview bytes.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`selection`](visuals/selection.md) | module | Deterministic bounded gallery selection |
| [`dataset`](visuals/dataset.md) | module | Dataset preview-gallery rendering |
| [`predictions`](visuals/predictions.md) | module | Prediction-overlay visuals over frozen galleries (classifier) |

## Package interface

`tools.ml_models.analysis.visuals.__init__` carries a module docstring
only.

## Interactions

`predictions` renders `PredictionGallery` selections produced by
`analysis.prediction_selections` against capture-supplied preview
NPZ bytes; segmentation visuals remain deferred to PR14.

## Constraints

- Visuals consume captured evidence; they never rerun inference.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
