# sim.environment.look

**Source:** `packages/sim/src/sim/environment/look.py`
**Kind:** module

## Purpose

The look module computes mount look angles from ISS ECI state to an ECEF CoG.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `look_angles_at` | function | ISS plus CoG to `LookAngles` |
| `earth_occludes_cog` | function | True when a height-0 Earth hit is closer than the CoG |

## Inputs and outputs

**`look_angles_at(iss, cog_ecef_m, earth, omega_earth_rad_s, epoch_utc_s) -> LookAngles`**

- Inputs: `IssState`, ECEF CoG meters, `EarthModel`, Earth rate, epoch.
- Output: azimuth, elevation, off-nadir, slant, incidence, visibility.

**`earth_occludes_cog(iss, cog_ecef_m, earth, omega_earth_rad_s, epoch_utc_s) -> bool`**

- Inputs: `IssState`, ECEF CoG meters, `EarthModel`, Earth rate, epoch.
- Output: True when the nearest height-0 Earth hit is closer than the CoG.

## Behavior

1. The CoG rotates into ECI with `eci_from_ecef`.
2. LVLH axes come from flight `lvlh_axes`.
3. Elevation is `atan2(along, nadir)` (0 at nadir).
4. Visibility uses `earth.intersect_at_height` at height 0 m.
5. A CoG behind that hit is not visible. A ray that misses Earth is not
   occlusion.

## Errors and faults

None. A near-zero slant returns `visible=False` and zero angles.

## Messages

None.

## Configuration

None.

## Constraints

Angles are radians. Slant is meters. This is not the study km/deg Look type.

## Related documents

- [`sim.environment.records`](records.md)
- [`sim.environment.evaluate`](evaluate.md)
