# flight.payload.gimbal.pointing

**Source:** `packages/flight/src/flight/payload/gimbal/pointing.py`
**Kind:** pure module

## Purpose

This module converts blob centroids in band-plane pixel space to pinhole boresight
error. Image `+x` is unactuated optical azimuth. Image `+y` (down) is `-elevation`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `pinhole_error_rad` | function | Returns `(e_az_rad, e_el_rad)` from boresight |
| `boresight_error_deg` | function | Same error in degrees |
| `target_displacement_px` | function | Euclidean band-plane distance from boresight |

## Inputs and outputs

Functions take `centroid_px`, band-plane width and height, band pixel pitch in
meters, and focal length in meters. `target_displacement_px` does not use optics
scalars.

## Behavior

1. Build a pinhole camera ray through the centroid.
2. Elevation error is `-atan2(d_cam_y, d_cam_z)`. Azimuth error is
   `atan2(d_cam_x, d_cam_z)` and is unactuated.
3. Displacement is `hypot(dx, dy)` in band-plane pixels.

## Errors and faults

None.

## Messages

None.

## Configuration

Band-plane size is `SensorConfig` mosaic size divided by two. Band pitch is
`2 * pixel_um`. Focal length is `focal_length_mm`.

## Constraints

The functions assume the inference tensor is the full band plane with no crop and no
scale. Error is pinhole geometry, not `px * IFOV`.

## Related documents

- [`flight.payload.gimbal.geo`](geo.md)
- [`flight.payload.gimbal.safety`](safety.md)
- [`flight.payload.control`](../control.md)
