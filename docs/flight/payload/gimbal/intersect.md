# flight.payload.gimbal.intersect

**Source:** `packages/flight/src/flight/payload/gimbal/intersect.py`
**Kind:** pure module

## Purpose

Each accepted vision frame rebuilds the CoG pinhole ray, rotates it into ECI, and
intersects a constant-height WGS-84 ellipsoid. The hit is stored in ECEF meters.
`intersect_boresight` uses the principal-point ray at the current elevation.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `IntersectResult` | dataclass | ECEF hit, hit flag, and slant range |
| `intersect_cog` | function | Intersect one CoG ray with the height ellipsoid |
| `intersect_boresight` | function | Intersect the principal-point ray with the height ellipsoid |

## Inputs and outputs

`intersect_cog` takes the band-plane centroid, gimbal elevation, ISS ECI state,
Earth-rotation scalars, pinhole geometry, the last CoG, and `height_m`. It returns
`IntersectResult`. A miss keeps the last CoG and sets `hit=False`.

`intersect_boresight` takes gimbal elevation, ISS ECI state, Earth-rotation
scalars, the last CoG, and `height_m`. It does not take a pixel centroid. A miss
keeps the last CoG.

## Behavior

1. Build the camera ray from the centroid, or use mount boresight at `theta_g`.
2. Rotate the ray into ECI and ECEF.
3. Intersect the height ellipsoid. On a hit, return the ECEF point. On a miss,
   return the last CoG.

## Errors and faults

None. A miss is `hit=False`, not a fault.

## Messages

None.

## Configuration

WGS-84 scalars and Earth rate come from `EphemerisConfig`. Pixel pitch and focal
length come from `SensorConfig`. `height_m` comes from
`ControllerConfig.predictor.cog_height_m`.

## Constraints

The function is pure. Callers must not finite-difference successive intersects for
the co-rotating rate. The height offset is a tracking proxy. It is not a
plume-physics height.

## Related documents

- [`flight.payload.gimbal.geo`](geo.md)
- [`flight.payload.gimbal.predictor`](predictor.md)
- [`flight.payload.control`](../control.md)
