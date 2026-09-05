# flight.payload.gimbal.predictor

**Source:** `packages/flight/src/flight/payload/gimbal/predictor.py`
**Kind:** pure module

## Purpose

The predictor returns signed off-nadir elevation of a frozen ECEF CoG and the
co-rotating elevation rate `omega_t_nom`. It does not finite-difference successive
intersects.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `predict_los` | function | `(theta_los_rad, omega_t_nom_rad_s)` |

## Inputs and outputs

Inputs are UTC seconds, ISS ECI position and velocity, the frozen ECEF CoG, Earth
rate, and the ECI/ECEF alignment epoch. Outputs are elevation and elevation rate.

## Behavior

1. Rotate the frozen ECEF CoG into ECI at the current UTC.
2. Project the look vector onto LVLH `+y` and `+z` and form `atan2`.
3. Differentiate that elevation with Earth rotation and ISS motion through an
   analytic Jacobian.

## Errors and faults

None. A degenerate range returns rate `0.0`.

## Messages

None.

## Configuration

Earth rate and epoch come from `EphemerisConfig`.

## Constraints

The CoG stays fixed in ECEF for this call. Walk between vision frames is a new
intersect, not a slope inside this function.

## Related documents

- [`flight.payload.gimbal.geo`](geo.md)
- [`flight.payload.gimbal.intersect`](intersect.md)
- [`flight.payload.gimbal.outer`](outer.md)
