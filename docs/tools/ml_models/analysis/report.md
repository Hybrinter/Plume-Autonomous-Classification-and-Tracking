# tools.ml_models.analysis.report

**Source:** `packages/tools/src/tools/ml_models/analysis/report.py`
**Kind:** module

## Purpose

This module writes `figures/` PNGs and `report.md` into a training run
directory from its history, evaluation, and prediction artifacts.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `write_report` | function | Emit figures and `report.md` for one run directory |

## Inputs and outputs

`write_report(run_dir) -> Path` returns the written `report.md` path.

## Behavior

1. Builds history, overlay, and failure figures through
   `tools.ml_models.analysis.plots`.
2. Reads `summary.json` and `eval.json` when present and renders kind,
   architecture, best epoch, validation and test sections.
3. Links written figures from the report.

## Errors and faults

`FileNotFoundError` when the run directory is missing. JSON decode errors
surface unchanged.

## Messages

None.

## Configuration

None.

## Constraints

- The report reflects stored artifacts; it does not recompute metrics.
- `report.md` is rewritten on each call.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.plots`](plots.md)
- [`tools.ml_models.analysis.runs`](runs.md)
