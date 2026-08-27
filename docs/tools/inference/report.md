# tools.inference.report

**Source:** `packages/tools/src/tools/inference/report.py`
**Kind:** module

## Purpose

This module writes figures and a markdown summary into a training run directory.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `write_report` | function | Emit `figures/` PNGs and `report.md` |

## Inputs and outputs

`write_report(run_dir) -> Path`. Returns the `report.md` path.

## Behavior

1. Build history curves, overlays, and a failure gallery when files exist.
2. Save PNGs under `run_dir/figures/`.
3. Write `report.md` with summary fields, eval metrics, and figure links.

## Errors and faults

`FileNotFoundError` when the run directory is missing.

## Messages

None.

## Configuration

None.

## Constraints

This module does not import `tools.analysis`. Missing history or eval files skip
those sections.

## Related documents

- [`tools.inference.plots`](plots.md)
- [`tools.inference.eval`](eval.md)
- [`tools.inference.train`](train.md)
