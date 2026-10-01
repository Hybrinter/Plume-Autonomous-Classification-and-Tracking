# tools.ml_models.dataset.sources.zenodo.adapt

**Source:** `packages/tools/src/tools/ml_models/dataset/sources/zenodo/adapt.py`
**Kind:** module

## Purpose

This module is the `ZenodoSource`: it reads every labeled Zenodo 4250706
image once from the archive and emits each requested GSD-bin variant as a
`RawTile`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ZenodoSource` | class | `RawSource` over the image and annotation archives |

## Inputs and outputs

`ZenodoSource(images_tar, labels_tar, weight_table, bins=DEFAULT_BINS)`
indexes the archives and precomputes one `RawTileRef` per `(tile, bin)`.
`index()` returns the refs in image-archive order with `bin_id` set,
`group_id` equal to `location_id`, `label` from the class directory, and
`has_mask` from annotation presence. `iter_tiles()` yields `RawTile` rows
in index order: prism-mixed, area-resampled `(3, h, w)` unit images plus a
rasterized mask when the tile is annotated.

## Behavior

1. The class directory supplies the presence label. Annotation presence,
   not the label, controls segmentor eligibility (`has_mask`).
2. All dates and bins at one location share `group_id`, so group splits
   never separate variants of a location.
3. Each image streams once; per bin the native reflectance is
   `resample_area`-downsampled and clipped to `[0, 1]`, and the mask is
   rasterized at the same output grid.
4. Stored GSD is `actual_gsd(bin)`: `1200 / (W, H)` of the rounded grid.

## Errors and faults

`ValueError` on an empty or duplicate-named `bins` table, a bin below the
native resolution, a malformed band order, an archive indexing failure, or
a stream that ends before the index. `FileNotFoundError` on a missing
archive. `ImportError` when rasterio is absent.

## Messages

None.

## Configuration

`weight_table` is a loaded `prism.WeightTable`; its `id` is exposed as
`weight_table_id` for `build_zenodo` provenance. `source_ref` is
`10.5281/zenodo.4250706`. `domain` is `unit`; `extent_m` is
`(1200, 1200)`.

## Constraints

`iter_tiles` is a single forward pass over the image archive. This module
does not import torch. Nothing extracts to disk.

## Related documents

- [`tools.ml_models.dataset.sources.zenodo`](../zenodo.md)
- [`tools.ml_models.dataset.sources.zenodo.archive`](archive.md)
- [`tools.ml_models.dataset.sources.zenodo.bins`](bins.md)
- [`tools.ml_models.dataset.raw`](../raw.md)
- [`tools.ml_models.dataset.build`](../build.md)
