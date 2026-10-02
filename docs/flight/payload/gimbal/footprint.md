# flight.payload.gimbal.footprint

**Source:** `packages/flight/src/flight/payload/gimbal/footprint.py`  
**Kind:** pure module

## Purpose

This module measures local ground sample distance (GSD) by intersecting pixel rays with an Earth ellipsoid. It produces per-pixel or per-tile lateral and along-track GSD for model conditioning.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `GSD_REFERENCE_M` | constant | Default model GSD reference, 15.87 m |
| `GsdPair` | dataclass | Lateral and along-track ground metres per pixel |
| `pixel_gsd` | function | Estimate local GSD from symmetric half-pixel rays |
| `tile_gsd_grid` | function | Estimate GSD at the center of each tile |
| `nominal_iss_state` | function | Construct an equatorial circular reference orbit |
| `to_model_gsd` | function | Encode positive GSD values as log ratios to a reference |

## Inputs and outputs

`pixel_gsd` takes a pixel coordinate, gimbal elevation, ISS ECI position and velocity, time and Earth scalars, camera geometry, and optional height. It returns `GsdPair` or `None` if the geometry is invalid or a ray misses the ellipsoid.

`tile_gsd_grid` takes the same geometry inputs plus a `(rows, cols)` grid. It returns float32 `(rows, cols, 2)` values ordered lateral then along-track, or `None` for invalid geometry, a grid that does not divide the camera frame, or any missed ray.

`nominal_iss_state(altitude_m, wgs84_a_m, mu_m3_s2)` returns ECI position, ECI velocity, and epoch for the equatorial circular reference orbit, or `None` for invalid scalars.

`to_model_gsd(gsd_m, reference_m)` accepts a nonempty array whose last axis has length two. It returns float32 `ln(GSD / reference)` values in `Ok`, or `Err(FRAME_MALFORMED)` for invalid or nonpositive values and invalid references.

## Behavior

1. `pixel_gsd` intersects four rays at half-pixel offsets around the requested coordinate.
2. It measures the two ECEF chord lengths and returns lateral then along-track GSD.
3. `tile_gsd_grid` evaluates each tile center in row-major order and stores each pair in `(rows, cols, 2)` order.
4. `to_model_gsd` computes log differences to avoid overflow in the ratio and rejects any non-finite encoded output.

## Errors and faults

`pixel_gsd`, `tile_gsd_grid`, and `nominal_iss_state` return `None` for invalid inputs or failed geometry. `to_model_gsd` returns `Err(FRAME_MALFORMED)` for invalid shape, non-finite or nonpositive GSD, invalid reference, or non-finite encoded output.

## Messages

None. This is a pure module.

## Configuration

The caller supplies sensor dimensions and optics through `CameraGeometry`, and ephemeris and Earth values through its scalar arguments. `GSD_REFERENCE_M` defaults to 15.87 m; `InferenceConfig.gsd_reference_m` supplies the configured reference to `to_model_gsd`.

## Constraints

All functions are pure and perform no I/O. GSD pairs are ordered lateral/W then along-track/H. The nominal state is a source-adaptation reference and does not replace measured ephemeris. Positive elevation looks forward; image-down rays look back, so along-track GSD decreases toward the bottom of the frame.

## Related documents

- [`flight.payload.gimbal.intersect`](intersect.md)
- [`flight.libs.config.config`](../../libs/config/config.md)
