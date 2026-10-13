# tools.ml_models.analysis.config

**Source:** `packages/tools/src/tools/ml_models/analysis/config.py`
**Kind:** module
**Status:** stub

## Purpose

This module declares the frozen configuration records for the analysis,
evaluation, and render boundaries.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `DatasetAnalysisConfig` | dataclass | `dataset` and `out` fields |
| `ModelAnalysisConfig` | dataclass | `run`, `out`, `checkpoint`, `final_test` fields |
| `EvaluationConfig` | dataclass | `kind`, `split`, `batch_size`, `device` fields |
| `PlotConfig` | dataclass | `formats` and `dpi` fields |

## Inputs and outputs

Constructor arguments only. Defaults: `checkpoint` is `"best"`,
`final_test` is `False`, `batch_size` is `2`, `device` is `"cpu"`,
`formats` is `("png", "svg")`, and `dpi` is `300`.

## Behavior

The records are frozen slots dataclasses. This scaffold declares the field
shapes and defaults only; strict validation lands in a later phase.

## Errors and faults

None at this layer.

## Messages

None.

## Configuration

None.

## Constraints

- `kind` values are `classifier` or `segmentor`; `split` values are
  `train`, `val`, or `test`.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.dataset`](dataset.md)
- [`tools.ml_models.analysis.model`](model.md)
