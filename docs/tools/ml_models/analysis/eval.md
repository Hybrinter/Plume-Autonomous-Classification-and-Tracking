# tools.ml_models.analysis.eval

**Source:** `packages/tools/src/tools/ml_models/analysis/eval.py`
**Kind:** module

## Purpose

This module scores a trained checkpoint on a named split and writes
`eval.json`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `evaluate` | function | Score a checkpoint and write eval artifacts |

## Inputs and outputs

`evaluate(run_dir, checkpoint=None, split="val", preview_limit=8) -> Path`.

The return value is `eval.json`. The call also writes `predictions.npz` and
updates `summary.json`.

## Behavior

1. Load `config.toml` and the processed pack used by the run.
2. Load the checkpoint and rebuild the graph from the registry.
3. Score the named split on torch tensors.
4. Store a preview of the lowest-scoring samples in `predictions.npz`.

## Errors and faults

`FileNotFoundError` when the run, checkpoint, or pack is missing. `ValueError`
on an unknown split name.

## Messages

None.

## Configuration

`split` is `train`, `val`, or `test`. The default is `val`. The preview stores
at most 8 samples.

## Constraints

This module imports torch. It does not import `flight.payload.inference`,
`flight.core`, or `tools.analysis`.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.report`](report.md)
- [`tools.ml_models.train`](../train.md)
