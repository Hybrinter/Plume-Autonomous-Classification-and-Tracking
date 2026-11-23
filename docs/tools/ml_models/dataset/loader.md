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
| `make_loader` | function | Seeded single-shard batches across one dataset's shards |

## Inputs and outputs

`ShardDataset(shard_dir, gsd_reference_m, task, *, channels)` memory-maps
`images.npy`, `gsd.npy`, `labels.npy`, and `masks.npy` when present.
`task` is `classifier` or `segmentor`; `channels` is a required keyword
and comes from the manifest band list.

`ShardDataset.__getitem__(index)` returns `(image, g, target)`. `image`
is an owned C-contiguous copy of the stored float32 unit `(C, H, W)` row,
checked to be finite and inside `[0, 1]`. `g` is float32 `(2,)`
from `to_model_gsd`. The classifier target is float32 `(1,)`; the segmentor
target is float32 `(1, H, W)`.

`make_loader(dataset, task, split, batch_size, seed, n_batches=n)` yields
`n_batches` stacked batches.

## Behavior

1. The manifest-listed `<task>/<split>` shards are opened sorted
   lexicographically by `<H>x<W>`; each `ShardDataset` length must equal
   the manifest's recorded row count.
2. Per batch, `numpy.random.default_rng(seed)` draws a shard with
   probability proportional to its row count.
3. Rows are drawn without replacement unless `batch_size` exceeds the
   shard size. The chosen indices form a `torch.utils.data.Subset`, and a
   single-batch `DataLoader` collates the batch. Every batch comes from a
   single shard, so H and W are uniform inside a batch.

## Errors and faults

`ValueError` on an unknown task, a non-positive `channels`, a stored array
with the wrong dtype or layout (`images.npy` float32 `(N, C, H, W)` with
positive dims, `gsd.npy` float32 `(N, 2)`, `labels.npy` float32 `(N, 1)`,
`masks.npy` uint8 `(N, 1, H, W)` when present), a stored image row outside
`[0, 1]`, a segmentor shard without `masks.npy`, a
`batch_size` or `n_batches` below 1, an empty `dataset` path, a shard
row count that disagrees with the manifest, or a dataset with no
shard for the requested task and split. `FileNotFoundError` when an array
file is missing.

## Messages

None.

## Configuration

`seed` seeds the NumPy Generator. There is no TOML file.

## Constraints

This is the only module in `tools.ml_models.dataset` that imports torch.
Shard arrays are read through `np.load` memmaps.

## Related documents

- [`tools.ml_models.dataset`](../dataset.md)
- [`tools.ml_models.dataset.store`](store.md)
- [`tools.ml_models.dataset.manifest`](manifest.md)
- [`tools.ml_models.dataset.gsd`](gsd.md)
