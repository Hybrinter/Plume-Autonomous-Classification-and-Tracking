# flight.payload.gimbal.pointing

**Source:** `packages/flight/src/flight/payload/gimbal/pointing.py`
**Kind:** pure module

## Purpose

The pointing module converts tensor pixel centroids to boresight-relative angular error and
full-plane pixel displacement.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `boresight_error_deg` | function | Returns (az_error_deg, el_error_deg) from boresight |
| `target_displacement_px` | function | Returns Euclidean distance from boresight in full-plane pixels |

## Inputs and outputs

Both functions take `centroid_px`, `crop_origin_px`, `scale_factor`, `plane_width_px`, and
`plane_height_px`. `boresight_error_deg` also takes `ifov_deg_per_px`.

Outputs are `(az, el)` degrees or a scalar displacement in pixels.

## Behavior

1. `_full_frame_px` inverts the crop and scale transform: full = crop_origin + tensor / scale.
2. Boresight sits at `(plane_width_px / 2, plane_height_px / 2)`.
3. Azimuth error is `(full_x - center_x) * ifov_deg_per_px`.
4. Elevation error is `-(full_y - center_y) * ifov_deg_per_px`.
5. Displacement is the hypotenuse of the offset in full-plane pixels.

Sign convention: image +x maps to +azimuth; image +y (down) maps to -elevation.

## Errors and faults

None.

## Messages

None.

## Configuration

Uses `SensorConfig.ifov_deg_per_px` and band-plane dimensions passed by the caller.

## Constraints

Pure module. `scale_factor` is always positive from preprocessing.

## Related documents

- [`flight.payload.gimbal`](gimbal.md)
- [`flight.payload.preprocess.crop`](preprocess/crop.md)
