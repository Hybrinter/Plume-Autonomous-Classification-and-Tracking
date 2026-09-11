# tools.inference.runs

**Source:** `packages/tools/src/tools/inference/runs.py`
**Kind:** module

## Purpose

This module discovers local training run directories and formats list and
compare tables.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `discover_runs` | function | Sorted run paths that contain `summary.json` |
| `load_summary` | function | `summary.json` plus eval overlay fields |
| `format_list` | function | Text table of discovered runs |
| `format_compare` | function | Side-by-side table of selected fields |
| `rank_runs` | function | Sort summaries by a val metric then FLOPs |
| `format_rank` | function | Compare table in ranked order |

## Inputs and outputs

`discover_runs(root) -> tuple[Path, ...]`.

`load_summary(run_dir) -> dict`.

`format_list(runs) -> str`.

`format_compare(runs) -> str`.

`rank_runs(runs, metric) -> tuple[dict, ...]`.

`format_rank(rows) -> str`.

## Behavior

1. Treat a directory as a run when it contains `summary.json`.
2. Overlay `eval.json` metric fields with a `val_` or `test_` prefix. Existing
   keys for the other split stay in `summary.json`.
3. Print tab-separated tables for the CLI.
4. `rank_runs` sorts by `val_<metric>` (or `best_val_metric`). `bce` is
   minimized. Other metrics are maximized. FLOPs break ties.

## Errors and faults

`FileNotFoundError` when `summary.json` is missing for `load_summary`.

## Messages

None.

## Configuration

Compare columns: run id, kind, arch, best epoch, val metric, parameter count,
FLOPs, val F1, val mean IoU, test F1, test mean IoU, test n, dataset hash.

List columns include `n_params` and `flops`.

## Constraints

Torch-free. Runs live under `artifacts/runs/` by default.

## Related documents

- [`tools.inference.train`](train.md)
- [`tools.inference.eval`](eval.md)
- [`tools.inference.cost`](cost.md)
- [`tools.inference.cli`](cli.md)
- [`tools.inference.sweep`](sweep.md)
