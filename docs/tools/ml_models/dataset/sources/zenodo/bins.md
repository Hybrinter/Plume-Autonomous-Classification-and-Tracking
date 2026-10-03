# tools.ml_models.dataset.sources.zenodo.bins

**Source:** `packages/tools/src/tools/ml_models/dataset/sources/zenodo/bins.py`
**Kind:** module

## Purpose

This module defines the Zenodo GSD bins: the native 10 m grid plus five
fixed target-GSD bins at 15, 20, 25, 30, and 35 metres per pixel. The
table is a literal constant; it depends on no camera, orbit, or flight
geometry.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `EXTENT_M` | constant | 1200 m ground window per bin |
| `NATIVE_SIDE` | constant | 120 px side of the native grid |
| `DEFAULT_BINS` | constant | Fixed `BinSpec` table, native plus `gsd15`..`gsd35` |
| `bin_hw` | function | `(H, W)` pixels covering `EXTENT_M` at a bin's GSD |
| `actual_gsd` | function | Stored metres per pixel after rounding |

## Inputs and outputs

`DEFAULT_BINS` holds `("native10", "gsd15", "gsd20", "gsd25", "gsd30",
"gsd35")` `BinSpec` rows with equal lateral and along-track target GSD.
`bin_hw(bin_spec)` returns `(round(1200/along), round(1200/lateral))`:
`(120, 120)` at native and `(80, 80)`, `(60, 60)`, `(48, 48)`, `(40, 40)`,
`(34, 34)` for the coarser targets. `actual_gsd(bin_spec)` returns the
`GsdPair` `1200 / (width, height)` of the rounded grid, so `gsd35`
records `1200 / 34` metres per pixel, not the nominal 35.

## Behavior

1. Bin pixel counts round the fixed ground window, so the stored GSD is
   `extent / pixels`, not the requested target.
2. Both bin dimensions must stay at or above 10 m per pixel: bins only
   retain or coarsen native data.
3. A rounded bin with zero pixels on either axis is rejected.

## Errors and faults

`ValueError` when a bin GSD is below 10 m or non-finite, or a rounded
bin is empty.

## Messages

None.

## Configuration

None. The bin table is a module constant.

## Constraints

This module does not import torch or any flight module. The bins are a
fixed literal table.

## Related documents

- [`tools.ml_models.dataset.sources.zenodo`](../zenodo.md)
- [`tools.ml_models.dataset.sources.zenodo.archive`](archive.md)
- [`tools.ml_models.dataset.sources.zenodo.adapt`](adapt.md)
- [`tools.ml_models.dataset.raw`](../../raw.md)
