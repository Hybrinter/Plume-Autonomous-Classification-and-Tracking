# tools.original_dataset_analysis

**Source:** `packages/tools/src/tools/original_dataset_analysis/`
**Kind:** package

## Purpose

This package reads the Zenodo 4250706 archives and prepares band subsets and
coarse tiles for a ShuffleNet classifier and a DilateNet segmentor.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`bands`](original_dataset_analysis/bands.md) | module | Sentinel-2 ids and subset selection |
| [`index`](original_dataset_analysis/index.md) | module | Archive index and GeoTIFF reads |
| [`split`](original_dataset_analysis/split.md) | module | Location-grouped splits |
| [`grid`](original_dataset_analysis/grid.md) | module | Coarsening and mask rasterization |

## Package interface

Import the public names from each module. The package ``__init__`` does not
re-export them.

## Interactions

The package reads local tar archives. It does not import ``flight`` or
``tools.inference``.

## Constraints

- Rasterio is imported inside ``iter_stacks``.
- Tests build synthetic archives and numpy tiles. They do not download Zenodo.

## Related documents

- [`tools`](../tools.md)
- [`tools.original_dataset_analysis.bands`](original_dataset_analysis/bands.md)
- [`tools.original_dataset_analysis.index`](original_dataset_analysis/index.md)
- [`tools.original_dataset_analysis.split`](original_dataset_analysis/split.md)
- [`tools.original_dataset_analysis.grid`](original_dataset_analysis/grid.md)
