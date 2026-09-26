# tools.ml_models.analysis.results

**Source:** `packages/tools/src/tools/ml_models/analysis/results.py`
**Kind:** module

## Purpose

This module writes the native and ground-sample markdown tables for the band
study.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `write_stub_tables` | function | Tables with a blank score column |
| `write_filled_tables` | function | Present scores filled, others blank |

## Inputs and outputs

`write_stub_tables(order, path) -> None`.

`write_filled_tables(order, scores, path, prevalence=None) -> None`.

## Behavior

1. List every native cell and every ground-sample cell for the band order.
2. Leave a score blank when the key is absent.
3. A 120 px score fills the native row and the matching ground-sample row.

## Errors and faults

None.

## Messages

None.

## Configuration

Cell lists come from `tools.ml_models.data.matrix`.

## Constraints

These functions do not train a model.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.data.matrix`](../data/matrix.md)
