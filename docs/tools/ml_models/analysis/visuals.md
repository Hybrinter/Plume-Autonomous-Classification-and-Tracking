# tools.ml_models.analysis.visuals

**Source:** `packages/tools/src/tools/ml_models/analysis/visuals/`
**Kind:** package

## Purpose

The visuals package assembles visual evidence: deterministic bounded
gallery selection and dataset preview-gallery rendering are
implemented; prediction overlays remain a scaffold.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`selection`](visuals/selection.md) | module | Deterministic bounded gallery selection |
| [`dataset`](visuals/dataset.md) | module | Dataset preview-gallery rendering |
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
