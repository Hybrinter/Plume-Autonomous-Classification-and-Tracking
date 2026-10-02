# tools.ml_models.studies.band_matrix

**Source:** `packages/tools/src/tools/ml_models/studies/band_matrix/`
**Kind:** package

## Purpose

This package reads the Zenodo 4250706 archives and prepares band subsets and
coarse tiles for a ShuffleNet classifier and a DilateNet segmentor.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`bands`](band_matrix/bands.md) | module | Sentinel-2 ids and subset selection |
| [`index`](band_matrix/index.md) | module | Archive index and GeoTIFF reads |
| [`cache`](band_matrix/cache.md) | module | Native stack memmap |
| [`split`](band_matrix/split.md) | module | Location-grouped splits |
| [`grid`](band_matrix/grid.md) | module | Coarsening and mask rasterization |
| [`moments`](band_matrix/moments.md) | module | Train-split band moments |
| [`dataset`](band_matrix/dataset.md) | module | Tile samples |
| [`models`](band_matrix/models.md) | module | ShuffleNet and DilateNet |
| [`metrics`](band_matrix/metrics.md) | module | Test-split scores |
| [`train`](band_matrix/train.md) | module | Shared training loop |
| [`sweep`](band_matrix/sweep.md) | module | Native training sweep |
| [`matrix`](band_matrix/matrix.md) | module | Native band matrix |
| [`plots`](band_matrix/plots.md) | module | Score bar charts |
| [`cli`](band_matrix/cli.md) | module | Cell list and completeness check |
| [`results`](band_matrix/results.md) | module | Blank result tables |

## Package interface

Import the public names from each module. The package ``__init__`` does not
re-export them.

## Interactions

The package reads local tar archives. It does not import ``flight`` or
``tools.analysis``.

## Constraints

- Rasterio is imported inside ``iter_stacks``.
- Tests build synthetic archives and numpy tiles. They do not download Zenodo.

## Related documents

- [`tools`](../../tools.md)
- [`tools.ml_models.studies.band_matrix.bands`](band_matrix/bands.md)
- [`tools.ml_models.studies.band_matrix.index`](band_matrix/index.md)
- [`tools.ml_models.studies.band_matrix.cache`](band_matrix/cache.md)
- [`tools.ml_models.studies.band_matrix.split`](band_matrix/split.md)
- [`tools.ml_models.studies.band_matrix.grid`](band_matrix/grid.md)
- [`tools.ml_models.studies.band_matrix.moments`](band_matrix/moments.md)
- [`tools.ml_models.studies.band_matrix.dataset`](band_matrix/dataset.md)
- [`tools.ml_models.studies.band_matrix.models`](band_matrix/models.md)
- [`tools.ml_models.studies.band_matrix.metrics`](band_matrix/metrics.md)
- [`tools.ml_models.studies.band_matrix.train`](band_matrix/train.md)
- [`tools.ml_models.studies.band_matrix.sweep`](band_matrix/sweep.md)
- [`tools.ml_models.studies.band_matrix.matrix`](band_matrix/matrix.md)
- [`tools.ml_models.studies.band_matrix.plots`](band_matrix/plots.md)
- [`tools.ml_models.studies.band_matrix.cli`](band_matrix/cli.md)
- [`tools.ml_models.studies.band_matrix.results`](band_matrix/results.md)
