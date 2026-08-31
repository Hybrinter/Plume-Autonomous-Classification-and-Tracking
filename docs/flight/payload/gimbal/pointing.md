# flight.payload.gimbal.pointing

**Source:** `packages/flight/src/flight/payload/gimbal/pointing.py`
**Kind:** pure module

## Purpose

This module converts blob centroids in model-input pixel space to boresight-relative
angular error in degrees, to an area-weighted center of mass, and to full-plane pixel
displacement.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `area_weighted_com_px` | function | Returns area-weighted (x, y) of blobs, or None |
| `boresight_error_deg` | function | Returns (az_error_deg, el_error_deg) from boresight |
| `target_displacement_px` | function | Returns Euclidean full-plane distance from boresight |

## Inputs and outputs

`area_weighted_com_px` takes a tuple of `BlobMeta` and returns `(x, y)` or None.

`boresight_error_deg` and `target_displacement_px` take `centroid_px`,
`crop_origin_px`, `scale_factor`, and band-plane width and height.
`boresight_error_deg` also takes `ifov_deg_per_px`.

`boresight_error_deg` returns `(az, el)` degree offsets. Positive azimuth is image +x;
positive elevation is image -y (upward).

`target_displacement_px` returns a float distance in full-plane pixels.

The live ingest path passes crop origin and scale from the inference result.

## Behavior

1. `area_weighted_com_px` weights each `centroid_raw` by `pixel_area` and divides by
   the total area.
2. Invert the crop and scale transform to full band-plane pixel coordinates.
3. Subtract the plane center `(width/2, height/2)` for boresight reference.
4. Multiply horizontal and vertical offsets by IFOV for angular error; negate vertical
   for elevation sign convention.
5. For displacement, compute `hypot(dx, dy)` in full-plane pixels.

## Errors and faults

None.

## Messages

None.

## Configuration

Uses band-plane dimensions from `SensorConfig` (`width_px // 2`, `height_px // 2`) and
`ifov_deg_per_px`. Crop origin and scale come from the inference result when a caller
passes them.

## Constraints

Displacement remains defined in full-plane pixels. The controller ingest passes the
inference crop origin and scale into `boresight_error_deg`.

## Related documents

- [`flight.payload.preprocess.crop`](../preprocess/crop.md)
- [`flight.payload.gimbal.safety`](safety.md)
- [`flight.payload.control`](../control.md)
