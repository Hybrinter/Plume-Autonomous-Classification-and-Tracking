# tools.original_dataset_analysis.grid

**Source:** `packages/tools/src/tools/original_dataset_analysis/grid.py`
**Kind:** module

## Purpose

This module coarsens a 120 by 120 reflectance stack on a 1.2 km tile and
rasterizes percentage-space polygons onto the same grid.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `EXTENT_M` | constant | 1200 metres |
| `NATIVE_SIDE` | constant | 120 pixels |
| `LEGAL_SIDES` | constant | 120, 80, 60, 40, 30 |
| `gsd_m` | function | Metres per pixel |
| `coarsen` | function | Area-weighted resample |
| `rasterize_mask` | function | Polygon fill |

## Inputs and outputs

`gsd_m(side_px) -> float`.

`coarsen(planes, side_px) -> np.ndarray` with shape ``(C, side, side)``.

`rasterize_mask(polygons, side_px, rule) -> np.ndarray` with shape
``(1, side, side)``.

## Behavior

1. Legal sides are the divisors of 1200 m that yield 10, 15, 20, 30, and 40 m.
2. ``coarsen`` averages source pixels with overlap weights. Side 120 is a copy.
3. ``rule="touch"`` marks a cell when any interior sample lies in a polygon.
4. ``rule="half"`` marks a cell when at least half of its samples lie in a polygon.

## Errors and faults

`ValueError` when the side is not legal, the source is not ``(C, 120, 120)``,
or the mask rule is unknown.

## Messages

None.

## Configuration

None.

## Constraints

The native grid is 120 pixels on a side. Other source shapes are refused.

## Related documents

- [`tools.original_dataset_analysis`](../original_dataset_analysis.md)
- [`tools.original_dataset_analysis.bands`](bands.md)
