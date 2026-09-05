# flight.payload.gimbal.intersect

**Source:** `packages/flight/src/flight/payload/gimbal/intersect.py`
**Kind:** pure module

## Purpose

Each accepted vision frame rebuilds the CoG pinhole ray, rotates it into ECI, and
intersects the WGS-84 ellipsoid. The hit is stored in ECEF meters.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `IntersectResult` | dataclass | ECEF hit, hit flag, and slant range |
| `intersect_cog` | function | Intersect one CoG ray with WGS-84 |

## Inputs and outputs

`intersect_cog` takes the band-plane centroid, gimbal elevation, ISS ECI state,
Earth-rotation scalars, pinhole geometry, and the last CoG. It returns
`IntersectResult`. A miss keeps the last CoG and sets `hit=False`.

## Behavior

1. Build the camera ray from the centroid.
2. Rotate the ray into the mount frame at `theta_g`, then into ECI and ECEF.
3. Intersect WGS-84. On a hit, return the ECEF point. On a miss, return the last CoG.

## Errors and faults

None. A miss is `hit=False`, not a fault.

## Messages

None.

## Configuration

WGS-84 scalars and Earth rate come from `EphemerisConfig`. Pixel pitch and focal
length come from `SensorConfig`.

## Constraints

The function is pure. Callers must not finite-difference successive intersects for
the co-rotating rate.

## Related documents

- [`flight.payload.gimbal.geo`](geo.md)
- [`flight.payload.gimbal.predictor`](predictor.md)
- [`flight.payload.control`](../control.md)
