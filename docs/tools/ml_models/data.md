# tools.ml_models.data

**Source:** `packages/tools/src/tools/ml_models/data/`
**Kind:** package

## Purpose

The data package reads and writes processed packs, Zenodo tiles, the prism
proxy, and flight-frame canvases.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`meta`](data/meta.md) | module | Dataset identity, provenance, and the pack hash |
| [`pack`](data/pack.md) | module | On-disk pack and in-memory concatenation |
| [`split`](data/split.md) | module | Group-wise train, val, and test indices |
| [`norm`](data/norm.md) | module | DN, unit-interval, and per-band z-score recipes |
| [`bands`](data/bands.md) | module | Sentinel-2 ids and subset selection |
| [`grid`](data/grid.md) | module | Legal-side coarsening and any-side resample |
| [`matrix`](data/matrix.md) | module | Native band matrix for the Zenodo study |
| [`zenodo`](data/zenodo.md) | module | Archive index, tile cache, and location splits |
| [`annotations`](data/annotations.md) | module | Polygon labels for the Zenodo corpus |
| [`fetch`](data/fetch.md) | module | Checksum status, download, and 4-band preprocess |
| [`moments`](data/moments.md) | module | Train-split per-band mean and standard deviation |
| [`prism`](data/prism.md) | module | AP-3200T weights and the 76 px proxy pack |
| [`augment`](data/augment.md) | module | Dihedral transforms and feathered paste |
| [`canvas`](data/canvas.md) | module | Flight-frame scenes and windows |

## Package interface

`tools.ml_models.data.__init__` carries a module docstring only. Callers import
each module by name.

## Interactions

`norm` calls `flight.payload.preprocess.normalize.normalize_dn`. `pack` calls
`meta` and `split` to write sidecars and to assign groups. `zenodo` calls
`split.assign_group_splits` for location ids. `fetch` calls `annotations` and
`train.recipe`. `prism` calls `zenodo`, `grid`, and `pack`. `canvas` calls
`augment.feather_paste`. No module publishes on the bus.

## Constraints

- `fetch` imports `train.recipe`, and `train.recipe` imports torch.
- Other modules in this package do not import torch.
- No module imports `flight.payload.inference`, `flight.core`, or
  `tools.analysis`.
- Arrays on disk are float32. Images are `(N, C, H, W)`. Masks are
  `(N, 1, H, W)`. Labels are `(N, 1)`.
- Rasterio is imported inside the GeoTIFF reader in `zenodo`.

## Related documents

- [`tools.ml_models`](../ml_models.md)
- [`tools.ml_models.data.meta`](data/meta.md)
- [`tools.ml_models.data.pack`](data/pack.md)
- [`tools.ml_models.data.split`](data/split.md)
- [`tools.ml_models.data.norm`](data/norm.md)
- [`tools.ml_models.data.bands`](data/bands.md)
- [`tools.ml_models.data.grid`](data/grid.md)
- [`tools.ml_models.data.matrix`](data/matrix.md)
- [`tools.ml_models.data.zenodo`](data/zenodo.md)
- [`tools.ml_models.data.annotations`](data/annotations.md)
- [`tools.ml_models.data.fetch`](data/fetch.md)
- [`tools.ml_models.data.moments`](data/moments.md)
- [`tools.ml_models.data.prism`](data/prism.md)
- [`tools.ml_models.data.augment`](data/augment.md)
- [`tools.ml_models.data.canvas`](data/canvas.md)
