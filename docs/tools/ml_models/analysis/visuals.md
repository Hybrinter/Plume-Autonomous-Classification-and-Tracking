# tools.ml_models.analysis.visuals

**Source:** `packages/tools/src/tools/ml_models/analysis/visuals/`
**Kind:** package
**Status:** stub

## Purpose

The visuals package is the scaffold for visual-evidence assembly:
deterministic selections, dataset-row visuals, and prediction overlays.
Every submodule is unimplemented and no visual is produced yet.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`selection`](visuals/selection.md) | module | Representative/failure selection (scaffold) |
| [`dataset`](visuals/dataset.md) | module | Dataset-row visuals (scaffold) |
| [`predictions`](visuals/predictions.md) | module | Prediction-overlay visuals (scaffold) |

## Package interface

`tools.ml_models.analysis.visuals.__init__` carries a module docstring
only.

## Interactions

None yet.

## Constraints

- Visuals consume captured evidence; they never rerun inference.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
