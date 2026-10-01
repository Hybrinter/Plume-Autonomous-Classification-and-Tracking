# flight.payload.gimbal.footprint

**Source:** `packages/flight/src/flight/payload/gimbal/footprint.py`
**Kind:** pure module

## Purpose

This module measures ground sample distance on the Earth ellipsoid: per-pixel
metres from half-pixel ray intersects, a per-tile GSD grid for one frame, a
nominal circular-orbit ISS state for source adaptation, and the shared
`ln(GSD / GSD_REFERENCE_M)` model encoding.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `GSD_REFERENCE_M` | constant | 15.87 m reference for the log encoding |
| `GsdPair` | dataclass | Metres per pixel, lateral then along-track |
| `pixel_gsd` | function | Footprint at one pixel via half-pixel rays |
| `tile_gsd_grid` | function | `(rows, cols, 2)` float32 GSD at tile centers |
| `nominal_iss_state` | function | Equatorial circular ECI orbit at epoch zero |
| `to_model_gsd` | function | `(.., 2)` metres to float32 log ratio, `Result` |

## Inputs and outputs

`pixel_gsd(u, v, theta_g_rad, r_iss_eci_m, v_iss_eci_m_s, utc_s, epoch_utc_s,
omega_earth_rad_s, wgs84_a_m, wgs84_f, camera, height_m=0.0)` returns a
`GsdPair` or `None`. `tile_gsd_grid` returns an `(rows, cols, 2)` float32
array or `None`. `nominal_iss_state(altitude_m, wgs84_a_m, mu_m3_s2)` returns
`(r_eci, v_eci, utc)` or `None`. `to_model_gsd(gsd_m, reference_m)` returns
`Ok` of a float32 `(.., 2)` array or `Err(FRAME_MALFORMED)`.

## Behavior

1. `pixel_gsd` intersects the four half-pixel neighbours of `(u, v)` through
   `intersect_cog` and measures the ECEF chord of each axis pair.
2. `tile_gsd_grid` evaluates `pixel_gsd` at each tile center of a uniform
   grid; the frame must divide evenly.
3. `nominal_iss_state` is a degraded-inference reference orbit. It is not
   measured ephemeris.
4. `to_model_gsd` is the single formula shared by dataset build and
   inference: `ln(value / reference)` as float32.

## Errors and faults

`pixel_gsd` returns `None` on non-finite scalars, non-positive camera or
orbit parameters, a degenerate orbit normal, a missed ray, or a non-positive
separation. `tile_gsd_grid` returns `None` on an empty or indivisible grid
or a missed `pixel_gsd`. `nominal_iss_state` returns `None` on non-finite
or non-positive arguments. `to_model_gsd` returns `Err(FRAME_MALFORMED)`
on a bad reference or a malformed `(.., 2)` input.

## Messages

None.

## Configuration

`CameraGeometry` and `EphemerisConfig` supply camera and Earth scalars.
`GSD_REFERENCE_M` is 15.87, matching `inference.gsd_reference_m`.

## Constraints

The module is pure: no I/O, no bus, no clock. It depends only on
`flight.libs.types`, `flight.payload.gimbal.intersect`, and NumPy.

## Related documents

- [`flight.payload.gimbal.intersect`](intersect.md)
- [`flight.payload.gimbal.geo`](geo.md)
- [`flight.libs.config`](../libs/config/config.md)
- [`tools.ml_models.dataset.preprocess`](../../../tools/ml_models/dataset/preprocess.md)
