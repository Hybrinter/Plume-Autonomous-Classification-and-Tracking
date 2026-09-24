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
| [`normalize`](original_dataset_analysis/normalize.md) | module | Train-split band moments |
| [`dataset`](original_dataset_analysis/dataset.md) | module | Tile samples |
| [`models`](original_dataset_analysis/models.md) | module | ShuffleNet and DilateNet |
| [`metrics`](original_dataset_analysis/metrics.md) | module | Test-split scores |
| [`train`](original_dataset_analysis/train.md) | module | Shared training loop |
| [`matrix`](original_dataset_analysis/matrix.md) | module | Native band matrix |
| [`plots`](original_dataset_analysis/plots.md) | module | Score bar charts |
| [`cli`](original_dataset_analysis/cli.md) | module | Cell list and completeness check |
| [`results`](original_dataset_analysis/results.md) | module | Blank result tables |

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
- [`tools.original_dataset_analysis.normalize`](original_dataset_analysis/normalize.md)
- [`tools.original_dataset_analysis.dataset`](original_dataset_analysis/dataset.md)
- [`tools.original_dataset_analysis.models`](original_dataset_analysis/models.md)
- [`tools.original_dataset_analysis.metrics`](original_dataset_analysis/metrics.md)
- [`tools.original_dataset_analysis.train`](original_dataset_analysis/train.md)
- [`tools.original_dataset_analysis.matrix`](original_dataset_analysis/matrix.md)
- [`tools.original_dataset_analysis.plots`](original_dataset_analysis/plots.md)
- [`tools.original_dataset_analysis.cli`](original_dataset_analysis/cli.md)
- [`tools.original_dataset_analysis.results`](original_dataset_analysis/results.md)
