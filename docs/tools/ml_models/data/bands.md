# tools.ml_models.data.bands

**Source:** `packages/tools/src/tools/ml_models/data/bands.py`
**Kind:** module

## Purpose

This module maps GeoTIFF band descriptions to Sentinel-2 ids and selects a
named subset. `ZENODO_BAND_IDS` is the 13-band corpus order, with B10 last.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ZENODO_BAND_IDS` | constant | B1 through B12 with B10 last |
| `BandOrder` | class | Verified file order |
| `BandSpec` | class | Subset request |
| `BandSubset` | class | Selected ids and file indices |
| `coerce_descriptions` | function | Fill the corpus order when descriptions are empty |
| `verify_band_order` | function | Descriptions to a `BandOrder` |
| `resolve_subset` | function | A `BandSpec` against a verified order |

## Inputs and outputs

`coerce_descriptions(descriptions) -> tuple[str, ...]`.

`verify_band_order(descriptions) -> BandOrder`.

`resolve_subset(order, spec) -> BandSubset`.

## Behavior

1. Thirteen empty descriptions become B1, B2, B3, B4, B5, B6, B7, B8, B8A,
   B9, B11, B12, B10. Any present description is scanned for a Sentinel-2
   token. `B08` and `B8A` normalize to `B8` and `B8A`.
2. Duplicate ids and a missing `B10` token raise.
3. `rgb` selects B2, B3, B4. `s2_12` selects every id except B10.
   `loo` removes one id from that 12-band set. B10 is not a trainable input.
4. Selected indices follow file order.

## Errors and faults

`ValueError` on an empty description list of the wrong length, a duplicate id,
a missing B10 token, an unknown subset kind, or a leave-one-out id outside
the 12-band set.

## Messages

None.

## Configuration

None.

## Constraints

Subset selection uses verified ids. It does not assume a fixed raster index
for B10.

## Related documents

- [`tools.ml_models.data`](../data.md)
- [`tools.ml_models.data.matrix`](matrix.md)
- [`tools.ml_models.data.zenodo`](zenodo.md)
