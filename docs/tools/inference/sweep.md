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
| `sweep` | function | Train, eval val, and write `sweep.jsonl` |

## Inputs and outputs

`load_sweep_space(path) -> tuple[TrainConfig, ...]`.

`sweep(space_path, out=None) -> Path`. Returns the JSONL path. Default output is
`<run_dir>/sweep.jsonl` from the first trial.

Each JSONL record holds `run_id`, `path`, `kind`, `arch`, `learning_rate`,
`seed`, `optimizer`, `n_params`, `flops`, `val_metric`, `best_val_metric`, and
`val_*` summary fields.

## Behavior

1. Read the space TOML. Keys match `TrainConfig` fields plus `max_runs`.
2. Treat list values as search axes. Treat scalars as fixed fields.
3. Expand the cartesian product in sorted axis-name order. Truncate to
   `max_runs` when that value is positive.
4. Reject a scalar `run_id` and an unknown architecture pair.
5. Call `train` then `evaluate(..., split="val")` for each trial.
6. Write one JSON object per line to the JSONL path.

## Errors and faults

`ValueError` on an unknown key, empty axis, scalar `run_id`, empty trial list,
or unknown architecture. `FileExistsError` when a trial run directory exists.

## Messages

None.

## Configuration

Space keys overlay `TrainConfig`. `max_runs` is not a train field.

## Constraints

The sweep scores the val split. A later `eval --split test` scores the holdout.
Axis names sort with Python string order. Adding an architecture is one builder,
one `build()` branch, one `known()` pair, and one test.

## Related documents

- [`tools.inference.train`](train.md)
- [`tools.inference.eval`](eval.md)
- [`tools.inference.runs`](runs.md)
- [`tools.inference.arch.registry`](arch/registry.md)
- [`tools.inference.cli`](cli.md)
