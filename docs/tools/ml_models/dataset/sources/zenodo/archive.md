# tools.ml_models.dataset.sources.zenodo.archive

**Source:** `packages/tools/src/tools/ml_models/dataset/sources/zenodo/archive.py`
**Kind:** module

## Purpose

This module indexes the Zenodo 4250706 image and annotation tarballs and
streams GeoTIFF stacks in one forward pass. Nothing extracts to disk.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `TileRef` | dataclass | One corpus image: stem, location, label, member, polygons |
| `TileIndex` | dataclass | Stems in archive order, `by_stem` lookup |
| `location_id_of` | function | Leading underscore token of a stem |
| `build_index` | function | Index image stems, presence labels, and polygons |
| `iter_stacks` | function | Yield `(tile, stack, descriptions)` in archive order |
| `to_native_stack` | function | Pad or crop a `(C, H, W)` stack to `(C, 120, 120)` |

## Inputs and outputs

`build_index(images_tar, labels_tar) -> TileIndex` reads the Label Studio
JSON archive, then the image archive. `iter_stacks(images_tar, tiles)`
yields `(tile, float32 (C, H, W) stack, band descriptions)` in archive
order. `to_native_stack(stack)` returns float32 `(C, 120, 120)`.

## Behavior

1. Annotation members ending in `.json` map to image stems by stripping
   `_features.json` or the file suffix; polygons come from
   `annotations.parse_polygons`.
2. Image members ending in `.tif` or `.tiff` must sit under exactly one of
   a `positive` or `negative` archive directory.
3. Duplicate image stems keep the first member. A stem without an
   underscore token uses the whole stem as `location_id`.
4. `iter_stacks` opens the archive in streaming mode, yields each wanted
   member once, and stops when every tile is served.
5. `to_native_stack` allows a 2-pixel slack per side, then edge-pads a
   short side and crops a long side from the origin.

## Errors and faults

`FileNotFoundError` on a missing archive or a requested member absent from
the image tar. `ValueError` on an empty location id, an image member
outside the positive/negative class directories, an invalid annotation
payload, or a stack too far from 120 px per side. `ImportError` when
rasterio is absent.

## Messages

None.

## Configuration

None. `NATIVE_SIDE` is shared with [`bins`](bins.md).

## Constraints

Streams are forward-only; `iter_stacks` consumes each member at most once.
This module does not import torch.

## Related documents

- [`tools.ml_models.dataset.sources.zenodo`](../zenodo.md)
- [`tools.ml_models.dataset.sources.zenodo.annotations`](annotations.md)
- [`tools.ml_models.dataset.sources.zenodo.bins`](bins.md)
- [`tools.ml_models.dataset.sources.zenodo.adapt`](adapt.md)
