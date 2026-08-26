# tools.model.data

**Source:** `packages/tools/src/tools/model/data.py`
**Kind:** module

## Purpose

This module builds training batches as numpy arrays. It stays torch-free.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SampleBatch` | class | Images plus classifier or segmentor targets |
| `make_synthetic_batch` | function | Planted-blob synthetic scene |
| `load_disk_batch` | function | Packed `images.npy` plus labels or masks |

## Inputs and outputs

`make_synthetic_batch(kind, batch_size, channels, height, width, seed) -> SampleBatch`.

`load_disk_batch(data_dir, kind, bit_depth=12) -> SampleBatch`.

Images are `(N, C, H, W)` float32 in `[0, 1]`. Classifier targets are `(N, 1)`.
Segmentor targets are `(N, 1, H, W)`.

## Behavior

1. Synthetic even classifier samples get a bright rectangle and label 1.0.
2. Synthetic segmentor samples get the same rectangle as a mask.
3. Disk loader reads `images.npy` and `labels.npy` or `masks.npy`.
4. Values above 1.0 pass through `normalize_dn` with `bit_depth`.

## Errors and faults

`ValueError` on an unknown kind or a shape mismatch. `FileNotFoundError` on a
missing npy file.

## Messages

None.

## Configuration

`bit_depth` defaults to 12 for DN-valued disk images.

## Constraints

This module imports `flight.payload.preprocess.normalize_dn`. It does not import
torch.

## Related documents

- [`tools.model.train`](train.md)
- [`flight.payload.preprocess.normalize`](../../flight/payload/preprocess/normalize.md)
