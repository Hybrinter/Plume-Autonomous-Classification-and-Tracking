# tools.ml_models.analysis.report

**Source:** `packages/tools/src/tools/ml_models/analysis/report.py`
**Kind:** module

## Purpose

This module writes figures and `report.md` into one training run directory.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `write_report` | function | Emit `figures/` PNGs and `report.md` |

## Inputs and outputs

`write_report(run_dir) -> Path`. The return value is `report.md`.

## Behavior

1. Read `history.csv`, `predictions.npz`, `summary.json`, and `eval.json`
   when those files exist.
2. Draw history curves, overlays, and a failure gallery.
3. Write a markdown summary of the run identity and the last eval.

## Errors and faults

`FileNotFoundError` when the run directory is missing.

## Messages

None.

## Configuration

None.

## Constraints

Figure files come from `tools.ml_models.analysis.plots`.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.plots`](plots.md)
- [`tools.ml_models.analysis.eval`](eval.md)
