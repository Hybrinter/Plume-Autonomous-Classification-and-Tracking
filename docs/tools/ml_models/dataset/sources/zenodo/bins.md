# tools.ml_models.dataset.sources.zenodo.bins

**Source:** `packages/tools/src/tools/ml_models/dataset/sources/zenodo/bins.py`
**Kind:** module

## Purpose

This module defines the Zenodo GSD bins: the native 10 m grid plus one bin
per boresight flight elevation, computed from flight's optical and Earth
geometry at a reference orbit.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `EXTENT_M` | constant | 1200 m ground window per bin |
| `NATIVE_SIDE` | constant | 120 px side of the native grid |
| `FLIGHT_ELEVATIONS_DEG` | constant | `(5, 15, 25, 35, 45)` boresight elevations |
| `DEFAULT_BINS` | constant | `make_bins()` at the default reference orbit |
| `make_bins` | function | `BinSpec` tuple from sensor and ephemeris config |
| `bin_hw` | function | `(H, W)` pixels covering `EXTENT_M` at a bin's GSD |
| `actual_gsd` | function | Stored metres per pixel after rounding |

## Inputs and outputs

`make_bins(altitude_m=460_000, sensor=None, ephemeris=None)` returns
`("native10", "elevation5", ..., "elevation45")` `BinSpec` rows; the
elevation rows record the boresight `pixel_gsd` at the nominal ISS state.
`bin_hw(bin_spec)` returns `(round(1200/along), round(1200/lateral))`.
`actual_gsd(bin_spec)` returns the `GsdPair` `1200 / (width, height)` of
the rounded grid.

## Behavior

1. `nominal_iss_state` supplies the circular reference orbit; each
   elevation bin measures the boresight footprint through
   `pixel_gsd`.
2. Bin pixel counts round the fixed ground window, so the stored GSD is
   `extent / pixels`, not the requested target.
3. Both bin dimensions must stay at or above 10 m per pixel: bins only
   retain or coarsen native data.

## Errors and faults

`ValueError` when the reference orbit or camera geometry is invalid, a
boresight ray misses Earth, a bin GSD is below 10 m or non-finite, or a
rounded bin is empty.

## Messages

None.

## Configuration

`SensorConfig` supplies `width_px`, `height_px`, `pixel_um`, and
`optics.focal_length_mm`; `EphemerisConfig` supplies the Earth and rotation
scalars. `altitude_m` defaults to 460 km.

## Constraints

This module does not import torch. The bins are evaluated once at import
through `DEFAULT_BINS`.

## Related documents

- [`tools.ml_models.dataset.sources.zenodo`](../zenodo.md)
- [`tools.ml_models.dataset.sources.zenodo.archive`](archive.md)
- [`tools.ml_models.dataset.sources.zenodo.adapt`](adapt.md)
- [`flight.payload.gimbal.footprint`](../../../../../flight/payload/gimbal/footprint.md)
