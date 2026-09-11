# flight.payload.gimbal.predictor

**Source:** `packages/flight/src/flight/payload/gimbal/predictor.py`
**Kind:** pure module

## Purpose

The predictor returns signed off-nadir elevation of a frozen ECEF CoG, the
co-rotating elevation rate `omega_el`, and the unactuated optical-azimuth rate
`omega_az`. It does not finite-difference successive intersects.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `predict_los` | function | `(theta_el_rad, omega_el_rad_s, omega_az_rad_s)` |

## Inputs and outputs

Inputs are UTC seconds, ISS ECI position and velocity, the frozen ECEF CoG, Earth
rate, and the ECI/ECEF alignment epoch. Outputs are elevation, elevation rate, and
optical-azimuth rate.

## Behavior

1. Rotate the frozen ECEF CoG into ECI at the current UTC.
2. Project the look vector onto LVLH `+x`, `+y`, and `+z`.
3. Form elevation as `atan2(lx, lz)` and optical azimuth as
   `atan2(ly, hypot(lx, lz))`.
4. Differentiate both angles with Earth rotation and ISS motion through an
   analytic Jacobian. LVLH `y_hat` is treated as inertially fixed.

## Errors and faults

None. A degenerate range returns rates `0.0` and `0.0`.

## Messages

None.

## Configuration

Earth rate and epoch come from `EphemerisConfig`.

## Constraints

The CoG stays fixed in ECEF for this call. Walk between vision frames is a new
intersect, not a slope inside this function. `omega_el` includes Earth rotation in
the actuated elevation axis. It is not orbital mean motion. `omega_az` is the
unactuated lateral rate (equator cross-track Earth rotation at nadir). The
predictor never commands `omega_az`.

## Related documents

- [`flight.payload.gimbal.geo`](geo.md)
- [`flight.payload.gimbal.intersect`](intersect.md)
- [`flight.payload.gimbal.outer`](outer.md)
