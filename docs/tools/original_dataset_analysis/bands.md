# tools.original_dataset_analysis.bands

**Source:** `packages/tools/src/tools/original_dataset_analysis/bands.py`
**Kind:** module

## Purpose

This module maps GeoTIFF band descriptions to Sentinel-2 ids and selects a
named subset.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `BandOrder` | class | Verified file order |
| `BandSpec` | class | Subset request |
| `BandSubset` | class | Selected ids and file indices |
| `verify_band_order` | function | Descriptions to a `BandOrder` |
| `resolve_subset` | function | A `BandSpec` against a verified order |

## Inputs and outputs

`verify_band_order(descriptions) -> BandOrder`.

`resolve_subset(order, spec) -> BandSubset`.

## Behavior

1. Each description is scanned for a Sentinel-2 token. ``B08`` and ``B8A``
   normalize to ``B8`` and ``B8A``.
2. Duplicate ids and a missing ``B10`` token raise.
3. ``rgb`` selects B2, B3, B4. ``ceiling`` selects every id except B10.
   ``loo`` removes one ceiling id. ``s2_13`` keeps the file order, including B10.
4. Selected indices follow file order.

## Errors and faults

`ValueError` on an empty description, a duplicate id, a missing B10 token, an
unknown subset kind, or a leave-one-out id outside the ceiling.

## Messages

None.

## Configuration

None.

## Constraints

Subset selection uses verified ids. It does not assume a fixed raster index
for B10.

## Related documents

- [`tools.original_dataset_analysis`](../original_dataset_analysis.md)
- [`tools.original_dataset_analysis.grid`](grid.md)
