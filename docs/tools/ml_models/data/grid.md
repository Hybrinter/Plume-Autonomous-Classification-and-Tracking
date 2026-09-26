# tools.ml_models.data.grid

**Source:** `packages/tools/src/tools/ml_models/data/grid.py`
**Kind:** module

## Purpose

This module coarsens a 120 by 120 stack on a 1.2 km tile and rasterizes
percentage-space polygons. A separate resampler accepts any output side,
including 76.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `EXTENT_M` | constant | 1200 metres |
| `NATIVE_SIDE` | constant | 120 pixels |
| `LEGAL_SIDES` | constant | 120, 80, 60, 40, 30 |
| `gsd_m` | function | Metres per pixel for a legal side |
| `coarsen` | function | Area-weighted resample onto a legal side |
| `resample_area` | function | Area-weighted resample onto any side >= 1 |
| `rasterize_mask` | function | Polygon fill on a legal side |
| `rasterize_percent_mask` | function | Polygon fill on any side >= 1 |

## Inputs and outputs

`gsd_m(side_px) -> float`. The value is `1200 / side_px`.

`coarsen(planes, side_px) -> np.ndarray` with shape `(C, side, side)`.
`planes` is `(C, 120, 120)`.

`resample_area(planes, side_px) -> np.ndarray` with shape `(C, side, side)`.
`planes` is `(C, H, W)`.

`rasterize_mask(polygons, side_px, rule) -> np.ndarray` with shape
`(1, side, side)`.

`rasterize_percent_mask(polygons, side_px, rule="half") -> np.ndarray` with
shape `(1, side, side)`.

## Behavior

1. Legal sides are 120, 80, 60, 40, and 30. Those sides are 10, 15, 20, 30,
   and 40 metres on the 1.2 km tile. 76 is not a legal side.
2. `coarsen` checks the native shape and the legal side, then calls
   `resample_area`. Side 120 is a copy.
3. `resample_area` weights each source pixel by its overlap with each output
   bin. It accepts side 76 and any other int >= 1.
4. `rule="touch"` marks a cell when any interior sample lies in a polygon.
5. `rule="half"` marks a cell when at least half of its samples lie in a
   polygon. `rasterize_percent_mask` defaults to `half`.
6. `rasterize_mask` checks the legal side, then calls
   `rasterize_percent_mask`. The percent rasterizer does not check
   `LEGAL_SIDES`.

## Errors and faults

`ValueError` when `gsd_m` or `coarsen` receives a side outside `LEGAL_SIDES`,
when `coarsen` receives a source other than `(C, 120, 120)`, when
`resample_area` or `rasterize_percent_mask` receives a side below 1, or when
the mask rule is unknown.

## Messages

None.

## Configuration

None.

## Constraints

The study set stays `{120, 80, 60, 40, 30}`. `gsd_m(76)` raises.
`resample_area` and `rasterize_percent_mask` accept 76.

## Related documents

- [`tools.ml_models.data`](../data.md)
- [`tools.ml_models.data.bands`](bands.md)
- [`tools.ml_models.data.prism`](prism.md)
