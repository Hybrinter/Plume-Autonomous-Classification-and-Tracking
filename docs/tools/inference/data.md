# tools.inference.data

**Source:** `packages/tools/src/tools/inference/data.py`
**Kind:** module

## Purpose

This module builds training batches as CPU float32 torch tensors. On-disk packs
stay numpy `.npy` files with frozen splits.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SampleBatch` | class | Images plus classifier or segmentor targets |
| `ProcessedPack` | class | Memmap-backed pack with frozen splits and hash |
| `SplitDataset` | class | Torch Dataset over one named split |
| `make_synthetic_batch` | function | Planted-blob synthetic scene |
| `make_synthetic_pack` | function | Even-index blobs with masks and labels |
| `write_processed_pack` | function | Write npy tensors, splits, and dataset.json |
| `load_disk_batch` | function | Packed `images.npy` plus labels or masks |
| `load_processed_pack` | function | Memmap loader with hash check |
| `load_split` | function | One named split as a `SplitDataset` |

## Inputs and outputs

`make_synthetic_batch(kind, batch_size, channels, height, width, seed) -> SampleBatch`.

`make_synthetic_pack(n, channels, height, width, seed) -> (images, masks, labels)`.

`write_processed_pack(dest_dir, images, masks, labels, recipe, source_doi="") -> DatasetMeta`.

`load_disk_batch(data_dir, kind, bit_depth=12) -> SampleBatch`.

`load_processed_pack(data_dir, bit_depth=12) -> ProcessedPack`.

`load_split(data_dir, kind, split, bit_depth=12) -> SplitDataset`.

Images are `(N, C, H, W)` float32 in `[0, 1]` on CPU. Classifier targets are
`(N, 1)`. Segmentor targets are `(N, 1, H, W)`. `SplitDataset` items are one
sample: image `(C, H, W)` and a matching target.

## Behavior

1. Synthetic even classifier samples get a bright rectangle and label 1.0.
2. Synthetic segmentor samples get the same rectangle as a mask.
3. `make_synthetic_pack` plants blobs on even indices and derives labels from masks.
4. Disk loader reads `images.npy` and `labels.npy` or `masks.npy`.
5. Values above 1.0 pass through `normalize_dn` with `bit_depth`.
6. `load_split` indexes `train`, `val`, or `test` from `splits.json`.
7. `load_processed_pack` rejects a `dataset.json` hash that does not match the files.
8. `write_processed_pack` writes numpy `.npy` files from torch or numpy inputs.

## Errors and faults

`ValueError` on an unknown kind or a shape mismatch. `FileNotFoundError` on a
missing npy file.

## Messages

None.

## Configuration

`bit_depth` defaults to 12 for DN-valued disk images.

## Constraints

This module imports `flight.payload.preprocess.normalize_dn`. Pack images load
as a copy-on-write memmap when they already sit in `[0, 1]`.

## Related documents

- [`tools.inference.split`](split.md)
- [`tools.inference.train`](train.md)
- [`flight.payload.preprocess.normalize`](../../flight/payload/preprocess/normalize.md)
