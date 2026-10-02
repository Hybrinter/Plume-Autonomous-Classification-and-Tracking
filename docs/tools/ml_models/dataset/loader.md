# tools.ml_models.dataset.loader

**Source:** `packages/tools/src/tools/ml_models/dataset/loader.py`
**Kind:** module

## Purpose

This module exposes finished shards as torch datasets and yields seeded
batches that each come from a single shard.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `Batch` | alias | `(image, g, target)` tensor triple |
| `ShardDataset` | class | `torch.utils.data.Dataset` over one shard directory |
| `make_loader` | function | Seeded single-shard batches across datasets |

## Inputs and outputs

`ShardDataset(shard_dir, gsd_reference_m, task)` memory-maps
`images.npy`, `gsd.npy`, `labels.npy`, and `masks.npy` when present.
`task` is `classifier` or `segmentor`.

`ShardDataset.__getitem__(index)` returns `(image, g, target)`. `image`
is float32 unit `(3, H, W)` from `dequantize_unit`. `g` is float32 `(2,)`
from `to_model_gsd`. The classifier target is `(1,)`; the segmentor
target is `(1, H, W)`.

`make_loader(dests, task, split, batch_size, weights, seed, n_batches=n)`
yields `n_batches` stacked batches.

## Behavior

1. Each `dataset.json` is loaded and `check_compatible` requires shared
   bands, unit norm, and GSD reference across `dests`.
2. Under each `<dest>/<task>/<split>`, directories that parse as `<H>x<W>`
   become `ShardDataset`s.
3. Per batch, `numpy.random.default_rng(seed)` draws a dataset with
   probability proportional to `weights` (equal when None), then a shard
   with probability proportional to its row count.
4. Rows are drawn without replacement unless `batch_size` exceeds the
   shard size. The chosen indices form a `torch.utils.data.Subset`, and a
   single-batch `DataLoader` collates the batch. Every batch comes from a
   single shard, so H and W are uniform inside a batch.

## Errors and faults

`ValueError` on an unknown task, a segmentor shard without `masks.npy`, a
`batch_size` or `n_batches` below 1, an empty `dests`, incompatible
manifests, a non-positive or wrong-length `weights`, or a dataset with no
shard for the requested task and split. `FileNotFoundError` when an array
file is missing.

## Messages

None.

## Configuration

`weights` are relative dataset sampling weights. `seed` seeds the NumPy
Generator. There is no TOML file.

## Constraints

This is the only module in `tools.ml_models.dataset` that imports torch.
Shard arrays are read through `np.load` memmaps.

## Related documents

- [`tools.ml_models.dataset`](../dataset.md)
- [`tools.ml_models.dataset.store`](store.md)
- [`tools.ml_models.dataset.manifest`](manifest.md)
- [`tools.ml_models.dataset.preprocess`](preprocess.md)
