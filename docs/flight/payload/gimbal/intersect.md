# flight.payload.gimbal.intersect

**Source:** `packages/flight/src/flight/payload/gimbal/intersect.py`
**Kind:** pure module

## Purpose

Each accepted vision frame rebuilds the CoG pinhole ray, rotates it into ECI, and
intersects a 2 km height proxy ellipsoid. The hit is a `RayHit` in ECEF meters.
`intersect_boresight` uses the principal-point ray at the current elevation.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `CameraGeometry` | dataclass | Band-plane size, pitch, and focal length |
| `RayHit` | dataclass | ECEF hit point and slant range |
| `intersect_cog` | function | Intersect one CoG ray with the height-proxy ellipsoid |
| `intersect_boresight` | function | Intersect the principal-point ray with the height-proxy ellipsoid |

## Inputs and outputs

`intersect_cog` takes the band-plane centroid, gimbal elevation, ISS ECI state,
Earth-rotation scalars, `CameraGeometry`, and `height_m`. It returns `RayHit` or
`None`.

`intersect_boresight` takes gimbal elevation, ISS ECI state, Earth-rotation
scalars, and `height_m`. It does not take a pixel centroid. A miss is `None`.
Callers keep the last CoG.

## Behavior

1. Build the camera ray from the centroid, or use mount boresight at `theta_g`.
2. Rotate the ray into ECI and ECEF.
3. Intersect the height-proxy ellipsoid. On a hit, return the ECEF point and
   slant range. On a miss, return `None`.

## Errors and faults

None. A miss is `None`, not a fault.

## Messages

None.

## Configuration

WGS-84 scalars and Earth rate come from `EphemerisConfig`. Pixel pitch and focal
length come from `SensorConfig`. `height_m` comes from
`ControllerConfig.predictor.cog_height_m`.

## Constraints

The function is pure. Callers must not finite-difference successive intersects for
the co-rotating rate. The intersect uses a 2 km height proxy ellipsoid. It is
not a geodetic-height solver.

## Related documents

- [`flight.payload.gimbal.geo`](geo.md)
- [`flight.payload.gimbal.predictor`](predictor.md)
- [`flight.payload.control`](../control.md)
