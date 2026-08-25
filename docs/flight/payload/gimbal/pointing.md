# flight.payload.gimbal.pointing

**Source:** `packages/flight/src/flight/payload/gimbal/pointing.py`
**Kind:** pure module

## Purpose

This module converts blob centroids in model-input pixel space to boresight-relative
angular error in degrees and to full-plane pixel displacement for deadband checks.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `boresight_error_deg` | function | Returns (az_error_deg, el_error_deg) from boresight |
| `target_displacement_px` | function | Returns Euclidean full-plane distance from boresight |

## Inputs and outputs

Both functions take `centroid_px`, `crop_origin_px`, `scale_factor`, and band-plane
width and height. `boresight_error_deg` also takes `ifov_deg_per_px`.

`boresight_error_deg` returns `(az, el)` degree offsets. Positive azimuth is image +x;
positive elevation is image -y (upward).

`target_displacement_px` returns a float distance in full-plane pixels.

## Behavior

1. Invert the crop and scale transform to full band-plane pixel coordinates.
2. Subtract the plane center `(width/2, height/2)` for boresight reference.
3. Multiply horizontal and vertical offsets by IFOV for angular error; negate vertical
   for elevation sign convention.
4. For displacement, compute `hypot(dx, dy)` in full-plane pixels.

## Errors and faults

None.

## Messages

None.

## Configuration

Uses band-plane dimensions from `SensorConfig` (`width_px // 2`, `height_px // 2`) and
`ifov_deg_per_px`. Crop origin and scale come from the inference result.

## Constraints

Deadband thresholds are defined in full-plane pixels for consistency across search
(decimated) and tracking (cropped) modes.

## Related documents

- [`flight.payload.preprocess.crop`](../preprocess/crop.md)
- [`flight.payload.gimbal.safety`](safety.md)
