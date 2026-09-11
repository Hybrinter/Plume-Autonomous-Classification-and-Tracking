# tools.inference.sweep

**Source:** `packages/tools/src/tools/inference/sweep.py`
**Kind:** module

## Purpose

This module expands a TOML search space into train jobs. Each trial writes a
run directory, scores the val split, and appends one JSONL record.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `load_sweep_space` | function | Expand a space TOML into `TrainConfig` trials |
| `completed_run_ids` | function | Run identifiers already present in a JSONL |
| `sweep` | function | Train, eval val, and write `sweep.jsonl` |

## Inputs and outputs

`load_sweep_space(path) -> tuple[TrainConfig, ...]`.

`completed_run_ids(jsonl) -> frozenset[str]`. Failed trials are excluded.

`sweep(space_path, out=None, resume=False, data_dir=None, run_dir=None) -> Path`.
Returns the JSONL path. Default output is `<run_dir>/sweep.jsonl` from the first
trial. `data_dir` and `run_dir` overlay every trial. `resume=True` appends to an
existing JSONL and skips the identifiers `completed_run_ids` already holds.

Each JSONL record holds `run_id`, `path`, `kind`, `arch`, `learning_rate`,
`seed`, `optimizer`, `n_params`, `flops`, `val_metric`, `best_val_metric`, and
`val_*` summary fields.

## Behavior

1. Read the space TOML. Keys match `TrainConfig` fields plus `max_runs`.
2. Treat list values as search axes. Treat scalars as fixed fields.
3. Expand the cartesian product in sorted axis-name order. Truncate to
   `max_runs` when that value is positive.
4. Overlay each trial onto `TrainConfig` (unknown train keys fail the schema) and
   reject an unknown architecture pair.
5. Create an exclusive lock file beside the JSONL. The lock name is the JSONL
   path with `.lock` appended.
6. Skip any trial whose run identifier is already in the JSONL when `resume`
   is true.
7. Call `train` then `evaluate(..., split="val")` for each remaining trial.
   A trial that raises records an `error` field and the sweep continues.
8. Write one JSON object per line to the JSONL path, then remove the lock.

## Errors and faults

`ValueError` on an unknown key, empty axis, scalar `run_id`, empty trial list,
or unknown architecture. `FileExistsError` when the lock file already exists.

## Messages

None.

## Configuration

Space keys overlay `TrainConfig`. `max_runs` is not a train field.

## Constraints

The sweep scores the val split. A later `eval --split test` scores the holdout.
Axis names sort with Python string order. Adding an architecture is one builder,
one `build()` branch, one `known()` pair, and one test. A killed sweep can leave
the lock file in place. Delete that file before you start a new sweep on the
same output.

## Related documents

- [`tools.inference.train`](train.md)
- [`tools.inference.eval`](eval.md)
- [`tools.inference.runs`](runs.md)
- [`tools.inference.arch.registry`](arch/registry.md)
- [`tools.inference.cli`](cli.md)
