# tools.ml_models.train.evaluate

**Source:** `packages/tools/src/tools/ml_models/train/evaluate.py`
**Kind:** module

## Purpose

This module scores complete dataset splits without training resampling.
It reports per-dataset metrics, per-bin metrics, and a dataset-weighted
macro combination.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `evaluate` | function | Score one split across finished datasets |

## Inputs and outputs

`evaluate(model, dests, manifests, kind, split, batch_size, weights,
device) -> dict[str, object]`. The model receives `(image, gsd)` pairs
from a `DataLoader` over each `ShardDataset`.

The report carries `split`, `aggregation` (`dataset_weighted_macro`),
`datasets` (path, source, dataset_hash, metrics, bins), and `combined`
(all shared metric keys plus `n`).

## Behavior

1. Selected manifest shards for `(task, split)` stream through a
   `DataLoader` in file order; every row scores exactly once.
2. Each row also lands in the bucket named by its `bin_id`
   (`unbinned` when empty).
3. Classifier rows accumulate logits for `classifier_metrics`; segmentor
   rows accumulate per-image IoU, Dice, a blob-gate IoU at 0.55, and BCE.
4. `combined` weights each dataset's summary by the given weights;
   `n` sums the sample counts.
5. The model's training flag is saved and restored around evaluation.

## Errors and faults

`ValueError` on misaligned metadata, an empty selected split, a shard
row-count disagreement with the manifest, non-finite or misaligned model
outputs, or an empty evaluation sample set.

## Messages

None.

## Configuration

`batch_size`, `weights`, and `device` come from `TrainConfig` through the
training loop.

## Constraints

Evaluation runs under `torch.no_grad`. Weights are positive and
prevalidated by `TrainConfig`.

## Related documents

- [`tools.ml_models.train`](../train.md)
- [`tools.ml_models.train.metrics`](metrics.md)
- [`tools.ml_models.train.loop`](loop.md)
- [`tools.ml_models.dataset.loader`](../dataset/loader.md)
