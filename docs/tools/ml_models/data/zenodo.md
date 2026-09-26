# tools.ml_models.data.zenodo

**Source:** `packages/tools/src/tools/ml_models/data/zenodo.py`
**Kind:** module

## Purpose

This module indexes the Zenodo 4250706 archives, reads GeoTIFF members, stores
native stacks, and assigns each location id to one split.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `TileRef` | class | One image stem |
| `TileIndex` | class | The corpus index |
| `location_id_of` | function | Leading stem token |
| `build_index` | function | Stems, presence, and polygons |
| `iter_stacks` | function | One forward pass over requested GeoTIFF members |
| `TileCache` | class | Stem-addressed float32 stacks |
| `to_native_stack` | function | Pad or crop a near-native tile to 120 |
| `build_cache` | function | One forward pass into a memmap |
| `build_mask_cache` | function | Rasterize each annotated tile once |
| `open_cache` | function | Reopen a written cache |
| `SidePack` | class | One prepared ground-sample size |
| `prepare_side` | function | Resample the native cache once |
| `open_side_pack` | function | Open a prepared side pack |
| `SplitRecipe` | class | Seed and site fractions |
| `LocationSplit` | class | Site assignment and stems |
| `assign_location_splits` | function | One split name per location |

## Inputs and outputs

`build_index(images_tar, labels_tar) -> TileIndex`.

`iter_stacks(images_tar, tiles) -> Iterator[tuple[TileRef, np.ndarray, tuple[str, ...]]]`.

`build_cache(images_tar, tiles, path) -> TileCache`.

`TileCache.reader(tile) -> np.ndarray` with shape `(C, 120, 120)`.

`assign_location_splits(index, recipe) -> LocationSplit`.

## Behavior

1. Annotation JSON members ending in `_features.json` yield smoke polygons in
   percentage coordinates. A file with no polygon stores an empty tuple.
2. Image members under a path component named `positive` are marked positive.
   Duplicate stems keep the first image.
3. `iter_stacks` opens the image archive once as a forward stream. Yield order
   follows the archive. A missing description is an empty string.
4. A tile whose sides are within 2 pixels of 120 is edge-padded or cropped to
   `(C, 120, 120)`. `build_cache` stores those stacks in `stacks.dat`.
5. `prepare_side` writes image, mask, presence, and annotation arrays for one
   legal side.
6. `assign_location_splits` passes unique location ids to
   `assign_group_splits`. Fractions apply to sites. Tiles keep archive order
   inside each split.

## Errors and faults

`FileNotFoundError` when an archive, a member, or a sidecar is missing.
`ValueError` when a stem has an empty location id, a stack is far from 120,
a side is illegal, a fraction is not positive, the fractions do not sum to 1,
or fewer than three locations are present. `ImportError` when rasterio is
absent. `KeyError` when a stem was not stored. `json.JSONDecodeError` on a
malformed annotation.

## Messages

None.

## Configuration

Default split fractions are 0.70, 0.15, and 0.15.

## Constraints

Archives are read as streams. The module does not extract them to a directory.
Rasterio is imported inside the GeoTIFF reader. The module does not import
`flight.payload.inference`, `flight.core`, or `tools.analysis`. The split unit
is the location id.

## Related documents

- [`tools.ml_models.data`](../data.md)
- [`tools.ml_models.data.split`](split.md)
- [`tools.ml_models.data.grid`](grid.md)
- [`tools.ml_models.data.prism`](prism.md)
