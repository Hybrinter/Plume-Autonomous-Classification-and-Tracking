# sim.environment.models.earth

**Source:** `packages/sim/src/sim/environment/models/earth.py`
**Kind:** module

## Purpose

The earth module supplies the `EarthModel` Protocol and named ellipsoid and
sphere implementations.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `EarthModel` | Protocol | `intersect_at_height(origin, unit, height)` |
| `Wgs84Ellipsoid` | class | Flight WGS-84 ellipsoid with height-proxy inflation |
| `SphereEarth` | class | Sphere of radius `a_m` plus height |
| `geocentric_radius_m` | function | Geocentric radius at a geocentric latitude |

## Inputs and outputs

**`EarthModel.intersect_at_height(origin_ecef_m, unit_ecef, height_m)`**

- Inputs: ECEF origin metres, unit ECEF direction, height-proxy metres.
- Output: `(hit_ecef_m, slant_m)` or `None` on a miss.

**`geocentric_radius_m(a_m, f, lat_rad) -> float`**

- Inputs: semi-major metres, flattening, geocentric latitude radians.
- Output: geocentric radius metres.

## Behavior

1. `Wgs84Ellipsoid` calls `wgs84_intersect_at_height` with stored `a_m` and `f`.
2. `SphereEarth` calls `wgs84_intersect` with radius `a_m + height_m` and `f = 0`.

## Errors and faults

None. A miss returns `None`.

## Messages

None.

## Configuration

`build_models` copies `EphemerisConfig.wgs84_a_m` and `wgs84_f`. `sphere` uses
the mean of `a` and `b`.

## Constraints

The Protocol declares only `intersect_at_height`. Implementations are frozen
dataclasses.

## Related documents

- [`sim.environment.models`](../models.md)
- [`sim.environment.look`](../look.md)
