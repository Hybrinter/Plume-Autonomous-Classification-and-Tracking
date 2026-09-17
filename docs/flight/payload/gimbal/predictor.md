# flight.payload.gimbal.predictor

**Source:** `packages/flight/src/flight/payload/gimbal/predictor.py`
**Kind:** pure module

## Purpose

The predictor returns a `LosPrediction` for a frozen ECEF CoG: signed
off-nadir elevation, co-rotating elevation rate `omega_el`, and unactuated
optical-azimuth rate `omega_az`. It does not finite-difference successive
intersects.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `LosPrediction` | dataclass | Named elevation, elevation rate, and azimuth rate |
| `predict_los` | function | Returns `LosPrediction` for a frozen ECEF CoG |

## Inputs and outputs

Inputs are UTC seconds, ISS ECI position and velocity, the frozen ECEF CoG, Earth
rate, and the ECI/ECEF alignment epoch. Outputs are named fields
`elevation_rad`, `elevation_rate_rad_s`, and `azimuth_rate_rad_s`.

## Behavior

1. Rotate the frozen ECEF CoG into ECI at the current UTC.
2. Project the look vector onto LVLH `+x`, `+y`, and `+z`.
3. Form elevation as `atan2(lx, lz)` and optical azimuth as
   `atan2(ly, hypot(lx, lz))`.
4. Differentiate both angles with Earth rotation and ISS motion through an
   analytic Jacobian. LVLH `y_hat` is treated as inertially fixed
   (`y_dot = 0`).

## Errors and faults

None. A degenerate range returns both rates as `0.0`.

## Messages

None.

## Configuration

Earth rate and epoch come from `EphemerisConfig`.

## Constraints

The CoG stays fixed in ECEF for this call. Walk between vision frames is a new
intersect, not a slope inside this function. `omega_el` includes Earth rotation in
the actuated elevation axis. It is not orbital mean motion. `omega_az` is the
unactuated lateral rate (equator cross-track Earth rotation at nadir). The
predictor never commands `omega_az`. Callers read named `LosPrediction` fields.

## Related documents

- [`flight.payload.gimbal.geo`](geo.md)
- [`flight.payload.gimbal.intersect`](intersect.md)
- [`flight.payload.gimbal.scene`](scene.md)
- [`flight.payload.gimbal.outer`](outer.md)
