# tools.original_dataset_analysis.index

**Source:** `packages/tools/src/tools/original_dataset_analysis/index.py`
**Kind:** module

## Purpose

This module indexes the image and label tar archives and reads GeoTIFF members
in one forward pass.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `TileRef` | class | One image stem |
| `TileIndex` | class | The corpus index |
| `location_id_of` | function | Leading stem token |
| `build_index` | function | Stems, presence, and polygons |
| `iter_stacks` | function | One forward pass over requested GeoTIFF members |

## Inputs and outputs

`build_index(images_tar, labels_tar) -> TileIndex`.

`iter_stacks(images_tar, tiles) -> Iterator[tuple[TileRef, np.ndarray, tuple[str, ...]]]`.

## Behavior

1. Annotation JSON members ending in ``_features.json`` yield smoke polygons in
   percentage coordinates. A file with no polygon stores an empty tuple.
2. Image members under a path component named ``positive`` are marked positive.
3. Duplicate stems keep the first image.
4. ``iter_stacks`` opens the image archive once as a forward stream. It reads
   each requested member when the stream reaches it and returns float32 bands
   plus description strings. A missing description is an empty string. Yield
   order follows the archive.

## Errors and faults

`FileNotFoundError` when an archive or member is missing. `ValueError` when a
stem has an empty location id. `ImportError` when rasterio is absent.
`json.JSONDecodeError` on a malformed annotation.

## Messages

None.

## Configuration

None.

## Constraints

Archives are read as streams. The module does not extract them to a directory.

## Related documents

- [`tools.original_dataset_analysis`](../original_dataset_analysis.md)
- [`tools.original_dataset_analysis.split`](split.md)
