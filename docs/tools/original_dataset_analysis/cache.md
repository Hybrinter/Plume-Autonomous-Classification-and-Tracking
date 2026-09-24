# tools.original_dataset_analysis.cache

**Source:** `packages/tools/src/tools/original_dataset_analysis/cache.py`
**Kind:** module

## Purpose

This module stores native GeoTIFF stacks in one memmap so later epochs do not
reread the archive.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `TileCache` | class | Stem-addressed float32 stacks |
| `build_cache` | function | One forward pass into a memmap |
| `open_cache` | function | Reopen a written cache |

## Inputs and outputs

`build_cache(images_tar, tiles, path) -> TileCache`.

`open_cache(path) -> TileCache`.

`TileCache.reader(tile) -> np.ndarray` with shape ``(C, 120, 120)``.

## Behavior

1. ``build_cache`` reads the requested members once, in archive order.
2. The directory holds ``stacks.dat`` and ``meta.json``. The sidecar stores
   stems and band descriptions.
3. ``reader`` returns a copy of one row.

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
