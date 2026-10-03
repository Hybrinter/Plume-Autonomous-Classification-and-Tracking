# tools.ml_models.train.evaluate

**Source:** `packages/tools/src/tools/ml_models/train/evaluate.py`
**Kind:** module

## Purpose

This module scores complete dataset splits without training resampling.
It reports whole-split metrics and per-bin metrics for one finished
dataset.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `evaluate` | function | Score one split of a finished dataset |

## Inputs and outputs

`evaluate(model, dataset, manifest, kind, split, batch_size, device) ->
dict[str, object]`. The model receives `(image, gsd)` pairs from a
`DataLoader` over each `ShardDataset`.

The report carries `split`, `dataset`, `source`, `dataset_hash`,
`metrics` (all metric keys plus `n`), and `bins` (per-bin metric dicts).

## Behavior

1. Manifest-listed shards for `(task, split)` stream through a
   `DataLoader` in file order; every row scores exactly once.
2. Each row also lands in the bucket named by its `bin_id`
   (`unbinned` when empty).
3. Classifier rows accumulate logits for `classifier_metrics`; segmentor
   rows accumulate per-image IoU, Dice, a blob-gate IoU at 0.55, and BCE.
4. The model's training flag is saved and restored around evaluation.

## Errors and faults

`ValueError` on an empty selected split, a shard row-count disagreement
with the manifest, non-finite or misaligned model outputs, or an empty
evaluation sample set.

## Messages

None.

## Configuration

`batch_size` and `device` come from `TrainConfig` through the training
loop.

## Constraints

Evaluation runs under `torch.no_grad`.

## Related documents

- [`tools.ml_models.train`](../train.md)
- [`tools.ml_models.train.metrics`](metrics.md)
- [`tools.ml_models.train.loop`](loop.md)
- [`tools.ml_models.dataset.loader`](../dataset/loader.md)
