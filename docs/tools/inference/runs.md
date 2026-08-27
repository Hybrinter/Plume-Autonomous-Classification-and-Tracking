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

## Inputs and outputs

`discover_runs(root) -> tuple[Path, ...]`.

`load_summary(run_dir) -> dict`.

`format_list(runs) -> str`.

`format_compare(runs) -> str`.

## Behavior

1. Treat a directory as a run when it contains `summary.json`.
2. Overlay `eval.json` numeric fields with a `test_` prefix when the split is
   test.
3. Print tab-separated tables for the CLI.

## Errors and faults

`FileNotFoundError` when `summary.json` is missing for `load_summary`.

## Messages

None.

## Configuration

Compare columns: run id, kind, arch, best epoch, val metric, test F1, test
mean IoU, test n, dataset hash.

## Constraints

Torch-free. Runs live under `artifacts/runs/` by default.

## Related documents

- [`tools.inference.train`](train.md)
- [`tools.inference.eval`](eval.md)
- [`tools.inference.cli`](cli.md)
