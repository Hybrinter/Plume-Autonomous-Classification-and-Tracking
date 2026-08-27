# tools.inference.eval

**Source:** `packages/tools/src/tools/inference/eval.py`
**Kind:** module

## Purpose

This module scores a trained checkpoint on a named split and writes `eval.json`.
Importing the module does not import torch.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `evaluate` | function | Score `best.pt` (or an overlay path) and write eval artifacts |

## Inputs and outputs

`evaluate(run_dir, checkpoint=None, split="test", preview_limit=8) -> Path`.

Returns the `eval.json` path. Also writes `predictions.npz` and updates
`summary.json`.

## Behavior

1. Load `config.toml` and the processed pack used by the run.
2. Load the checkpoint, rebuild the graph from the registry, and run the split.
3. Write split metrics to `eval.json`.
4. Store a preview of lowest-scoring samples in `predictions.npz`.

## Errors and faults

`ImportError` when torch is not installed. `FileNotFoundError` when the run,
checkpoint, or pack is missing. `ValueError` on an unknown split name.

## Messages

None.

## Configuration

Default checkpoint is `checkpoints/best.pt`. Default split is `test`.

## Constraints

Torch imports are lazy inside `evaluate`. This module does not import
`tools.analysis`.

## Related documents

- [`tools.inference.train`](train.md)
- [`tools.inference.metrics`](metrics.md)
- [`tools.inference.report`](report.md)
