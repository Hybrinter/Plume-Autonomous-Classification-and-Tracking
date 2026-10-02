# tools.ml_models.dataset.store

**Source:** `packages/tools/src/tools/ml_models/dataset/store.py`
**Kind:** module

## Purpose

This module writes and reads one finished-dataset shard: preallocated
`.npy` arrays plus a `rows.jsonl` identity file.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `RowRecord` | class | One `rows.jsonl` object: ids, bin, and element |
| `ShardWriter` | class | Preallocated shard memmaps plus the row list |
| `read_rows` | function | Load `rows.jsonl` |
| `read_images` | function | Load `images.npy` |
| `read_gsd` | function | Load `gsd.npy` |
| `read_labels` | function | Load `labels.npy` |
| `read_masks` | function | Load `masks.npy`, or None when absent |

## Inputs and outputs

`ShardWriter(directory, count, height, width, *, channels,
with_masks=...)` creates the
directory and allocates `images.npy` float32 `(N, C, H, W)`, `gsd.npy`
float32 `(N, 2)`, `labels.npy` float32 `(N, 1)`, and `masks.npy` uint8
`(N, 1, H, W)` when `with_masks` is True. `channels` is a required
keyword.

`ShardWriter.append(image, gsd_m, label, mask, row)` writes the next row;
`image` is float32 `(C, H, W)`.
`ShardWriter.close()` flushes the arrays and writes `rows.jsonl`.
`ShardWriter.abort()` drops the memmap handles without writing rows.

The readers return `np.ndarray` arrays, or `tuple[RowRecord, ...]` for
`read_rows`.

## Behavior

1. The writer creates the shard directory and refuses an existing one.
2. `append` writes row `cursor` and advances it. A classifier shard rejects
   a mask; a segmentor shard requires one.
3. `close` raises when fewer rows were appended than allocated.
4. A shard directory is `<task>/<split>/<H>x<W>` under the dataset root.
   The caller renames the dataset root after every shard has closed.

## Errors and faults

`ValueError` on a non-positive shard shape, a full shard, a `channels`
below 1, or an array with
the wrong dtype or shape. `RuntimeError` when a classifier shard receives
a mask, a segmentor shard misses one, or `close` runs before the shard is
full. `read_rows` raises `ValueError` when a line is not an object with
exactly the row keys or a field has the wrong type.

## Messages

None.

## Configuration

Row keys are `tile_id`, `group_id`, `frame_id`, `grid_rc`, `bin_id`,
`element`, `theta_g_deg`, and `gsd_nominal`. Rows written before the
angle and nominal fields existed decode with `theta_g_deg` None and
`gsd_nominal` False. There is no TOML file.

## Constraints

Writes go through `numpy.lib.format.open_memmap` memmaps. `abort` releases
the handles so the temporary directory can be removed after a failed
build. This module does not import torch.

## Related documents

- [`tools.ml_models.dataset`](../dataset.md)
- [`tools.ml_models.dataset.build`](build.md)
- [`tools.ml_models.dataset.loader`](loader.md)
- [`tools.ml_models.dataset.manifest`](manifest.md)
