# flight.payload.gimbal.geo

**Source:** `packages/flight/src/flight/payload/gimbal/geo.py`
**Kind:** pure module

## Purpose

This module holds mount, camera, LVLH, and WGS-84 helpers for elevation pointing.
All functions are side-effect free. Angles are radians unless a name says otherwise.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `rx_neg` | function | Rotation matrix `R_x(-theta)` |
| `boresight_mount` | function | Unit boresight `(0, sin theta, cos theta)` |
| `rz` | function | Right-handed rotation about `+z` |
| `eci_from_ecef` | function | ECEF vector to ECI |
| `ecef_from_eci` | function | ECI vector to ECEF |
| `lvlh_axes` | function | ISS `+x` starboard, `+y` along-track, `+z` nadir |
| `pinhole_cam_ray` | function | Unit camera-frame ray from a band-plane pixel |
| `cam_ray_to_mount` | function | Camera ray into the mount frame |
| `mount_to_eci` | function | Mount vector into ECI via LVLH |
| `wgs84_intersect` | function | Forward ellipsoid intersect |

## Inputs and outputs

Vector inputs are ECI or ECEF meters as length-3 arrays. `wgs84_intersect` returns
`(hit_ecef_m, slant_m)` or `None` on a miss.

## Behavior

1. `lvlh_axes` builds `+x` from orbit angular momentum, `+z` as nadir, and `+y` as
   along-track.
2. `pinhole_cam_ray` places the principal point at the band-plane center.
3. `cam_ray_to_mount` maps camera `+Y` (image down) onto mount `-y` at nadir, then
   applies `R_x(-theta_g)`.
4. `wgs84_intersect` solves the ellipsoid quadratic and keeps the nearest forward hit.

## Errors and faults

None.

## Messages

None.

## Configuration

Callers pass WGS-84 `a` and `f` and Earth rate from `EphemerisConfig`.

## Constraints

Functions perform no I/O. The mount frame is identity versus ISS LVLH.

## Related documents

- [`flight.payload.gimbal.intersect`](intersect.md)
- [`flight.payload.gimbal.predictor`](predictor.md)
- [`flight.payload.gimbal.pointing`](pointing.md)
