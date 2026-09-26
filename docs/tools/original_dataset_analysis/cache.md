# tools.original_dataset_analysis.cache

**Source:** `packages/tools/src/tools/original_dataset_analysis/cache.py`
**Kind:** module

## Purpose

This module stores native GeoTIFF stacks in one memmap. Later epochs read the
memmap. Public names are re-exported from
[`tools.ml_models.data.zenodo`](../ml_models/data/zenodo.md).

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `TileCache` | class | Stem-addressed float32 stacks |
| `to_native_stack` | function | Pad or crop a near-native tile to 120 |
| `build_cache` | function | One forward pass into a memmap |
| `build_mask_cache` | function | Rasterize each annotated tile once |
| `open_cache` | function | Reopen a written cache |
| `SidePack` | class | One prepared ground-sample size |
| `prepare_side` | function | Resample the native cache once |
| `open_side_pack` | function | Open a prepared side pack |

## Inputs and outputs

`build_cache(images_tar, tiles, path) -> TileCache`.

`open_cache(path) -> TileCache`.

`TileCache.reader(tile) -> np.ndarray` with shape ``(C, 120, 120)``.

## Behavior

1. A tile whose sides are within 2 pixels of 120 is edge-padded or cropped
   to ``(C, 120, 120)``. ``build_cache`` then reads the requested members once.
2. The directory holds ``stacks.dat`` and ``meta.json``. The sidecar stores
   stems and band descriptions.
3. ``reader`` returns a copy of one row.
4. ``prepare_side`` writes ``images.dat``, ``masks.dat``, ``positive.dat``,
   ``annotated.dat``, and ``meta.json`` for one legal side. ``open_side_pack``
   returns those arrays as a ``SidePack``.

## Errors and faults

`ValueError` when the tile list is empty, a stack is not ``(C, 120, 120)``, or
the memmap size does not match the sidecar. `FileNotFoundError` when the
archive, a member, or the sidecar is missing. `KeyError` when a stem was not
stored.

## Messages

None.

## Configuration

None.

## Constraints

The cache is a local artifact. It is not a result table.

## Related documents

- [`tools.original_dataset_analysis`](../original_dataset_analysis.md)
- [`tools.original_dataset_analysis.index`](index.md)
