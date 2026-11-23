# tools.ml_models.dataset.sources.zenodo

**Source:** `packages/tools/src/tools/ml_models/dataset/sources/zenodo/`
**Kind:** package

## Purpose

The zenodo package adapts the Zenodo 4250706 Sentinel-2 smoke corpus into
the `RawSource` contract. Archives are indexed and streamed in place; no
member extracts to disk.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`archive`](zenodo/archive.md) | module | Tar index and forward GeoTIFF reads |
| [`bands`](zenodo/bands.md) | module | Sentinel-2 band identity and subset order |
| [`prism`](zenodo/prism.md) | module | AP-3200T weight table and color mix |
| [`resample`](zenodo/resample.md) | module | Area-overlap downsample of image planes |
| [`annotations`](zenodo/annotations.md) | module | Label Studio polygons and mask rasterize |
| [`bins`](zenodo/bins.md) | module | Native and fixed target-GSD bins |
| [`adapt`](zenodo/adapt.md) | module | `ZenodoSource` raw source |
| [`fetch`](zenodo/fetch.md) | module | Checksum manifest, download, and verify |

## Package interface

`tools.ml_models.dataset.sources.zenodo.__init__` carries a module docstring
only. Callers import `adapt` and friends by module name, or reach the
source through `build_zenodo`.

## Interactions

`adapt.ZenodoSource` wires the package together: `archive` indexes and
streams, `bands` verifies band order, `prism` mixes to BLUE/GREEN/RED,
`resample` shrinks planes to each `bins` geometry, and `annotations`
rasterizes masks. `build_zenodo` in
[`tools.ml_models.dataset.build`](../build.md) constructs the source from
CLI paths.

## Constraints

- No module in this package imports torch.
- `archive` imports rasterio inside its GeoTIFF reader only.
- Bins may only retain or coarsen native 10 m data.

## Related documents

- [`tools.ml_models.dataset.sources`](../sources.md)
- [`tools.ml_models.dataset.raw`](../raw.md)
- [`tools.ml_models.dataset.build`](../build.md)
