# tools.ml_models.analysis.runs

**Source:** `packages/tools/src/tools/ml_models/analysis/runs.py`
**Kind:** module

## Purpose

This module discovers local training run directories and formats list,
compare, and rank tables.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `discover_runs` | function | Sorted directories that contain `summary.json` |
| `load_summary` | function | `summary.json` plus eval overlay fields |
| `format_list` | function | Text table of discovered runs |
| `format_compare` | function | Side-by-side table of selected fields |
| `rank_runs` | function | Sort by a val metric, then by FLOPs |
| `format_rank` | function | Compare table in ranked order |

## Inputs and outputs

`discover_runs(root) -> tuple[Path, ...]`.

`load_summary(run_dir) -> dict`.

`rank_runs(runs, metric) -> tuple[dict, ...]`.

`format_list`, `format_compare`, and `format_rank` return text tables.

## Behavior

1. A run directory is a folder that contains `summary.json`.
2. `load_summary` copies numeric eval fields onto the summary with a split
   prefix.
3. `rank_runs` orders by the named val metric. `bce` is minimized. Other
   metrics are maximized. FLOPs break ties.

## Errors and faults

`FileNotFoundError` when `summary.json` is missing. `ValueError` when the
JSON root is not an object.

## Messages

None.

## Configuration

None.

## Constraints

Catalog text stays in this module. Flight and study frontiers live in
`tools.ml_models.analysis.pareto`.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.pareto`](pareto.md)
