# tools.inference.eval

**Source:** `packages/tools/src/tools/inference/eval.py`
**Kind:** module

## Purpose

This module scores a trained checkpoint on a named split and writes `eval.json`.
Forward and metrics stay on torch tensors.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `evaluate` | function | Score `best.pt` (or an overlay path) and write eval artifacts |

## Inputs and outputs

`evaluate(run_dir, checkpoint=None, split="val", preview_limit=8) -> Path`.

Returns the `eval.json` path. Also writes `predictions.npz` and updates
`summary.json`.

## Behavior

1. Load `config.toml` and the processed pack used by the run.
2. Load the checkpoint, rebuild the graph from the registry, and run the split.
3. Write split metrics to `eval.json`.
4. Store a preview of lowest-scoring samples in `predictions.npz`.

## Errors and faults

`FileNotFoundError` when the run, checkpoint, or pack is missing. `ValueError`
on an unknown split name.

## Messages

None.

## Configuration

Default checkpoint is `checkpoints/best.pt`. Default split is `val`. Pass
`split="test"` to score the held-out test split.

## Constraints

`predictions.npz` stores numpy arrays for the plot layer. This module does not
import `tools.analysis`.

## Related documents

- [`tools.inference.train`](train.md)
- [`tools.inference.metrics`](metrics.md)
- [`tools.inference.report`](report.md)
