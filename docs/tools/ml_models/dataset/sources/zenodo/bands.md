# tools.ml_models.dataset.sources.zenodo.bands

**Source:** `packages/tools/src/tools/ml_models/dataset/sources/zenodo/bands.py`
**Kind:** module

## Purpose

This module names the 13 Sentinel-2 bands in the Zenodo 4250706 GeoTIFF
order, verifies band descriptions, and resolves named subsets.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ZENODO_BAND_IDS` | constant | File order of the 13 corpus bands, B10 last |
| `BandOrder` | dataclass | Verified file order and id to index map |
| `BandSpec` | dataclass | Named subset request: `rgb`, `s2_12`, or `loo` |
| `BandSubset` | dataclass | Ids and file indices for one subset |
| `coerce_descriptions` | function | Fill the corpus order when descriptions are empty |
| `verify_band_order` | function | Descriptions to a `BandOrder` |
| `resolve_subset` | function | A `BandSpec` against a verified order |

## Inputs and outputs

`coerce_descriptions(descriptions)` returns the input, or the 13 corpus ids
when every description is empty. `verify_band_order(descriptions)` returns
a `BandOrder`. `resolve_subset(order, spec)` returns a `BandSubset` with
file-order `ids` and `indices`.

## Behavior

1. Each description is reduced to alphanumeric characters and matched to a
   canonical Sentinel-2 id, longest token first (`B8A` before `B8`).
2. `rgb` selects B2, B3, B4. `s2_12` drops B10. `loo` drops one named band
   from the 12-band set.
3. `coerce_descriptions` requires exactly 13 bands before it fabricates the
   corpus order.

## Errors and faults

`ValueError` when a description has no band id, an id is empty or repeats,
B10 is absent, an empty-description fill sees other than 13 bands, a subset
kind is unknown, a `loo` band is not in the 12-band set, or a selected id
is missing from the order.

## Messages

None.

## Configuration

None.

## Constraints

This module does not import torch or rasterio.

## Related documents

- [`tools.ml_models.dataset.sources.zenodo`](../zenodo.md)
- [`tools.ml_models.dataset.sources.zenodo.prism`](prism.md)
- [`tools.ml_models.dataset.sources.zenodo.adapt`](adapt.md)
