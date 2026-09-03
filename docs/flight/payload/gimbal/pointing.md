# flight.payload.gimbal.pointing

**Source:** `packages/flight/src/flight/payload/gimbal/pointing.py`
**Kind:** pure module

## Purpose

This module converts blob centroids in band-plane pixel space to boresight-relative
angular error in degrees and to band-plane pixel displacement from boresight.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `boresight_error_deg` | function | Returns (az_error_deg, el_error_deg) from boresight |
| `target_displacement_px` | function | Returns Euclidean band-plane distance from boresight |

## Inputs and outputs

Both functions take `centroid_px` and band-plane width and height.
`boresight_error_deg` also takes `ifov_band_deg_per_px`.

`boresight_error_deg` returns `(az, el)` degree offsets. Positive azimuth is image +x;
positive elevation is image -y (upward).

`target_displacement_px` returns a float distance in band-plane pixels.

## Behavior

1. Subtract the plane center `(width/2, height/2)` for boresight reference.
2. Multiply horizontal and vertical offsets by band IFOV for angular error; negate
   vertical for elevation sign convention.
3. For displacement, compute `hypot(dx, dy)` in band-plane pixels.

## Errors and faults

None.

## Messages

None.

## Configuration

Uses band-plane dimensions from `SensorConfig` (`width_px // 2`, `height_px // 2`) and
`ifov_band_deg_per_px`. Centroids are already in band-plane pixels.

## Constraints

The functions assume the inference tensor is the full band plane with no crop and no
scale.

## Related documents

- [`flight.payload.gimbal.safety`](safety.md)
- [`flight.payload.control`](../control.md)
