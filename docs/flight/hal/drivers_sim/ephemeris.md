# flight.hal.drivers_sim.ephemeris

**Source:** `packages/flight/src/flight/hal/drivers_sim/ephemeris.py`
**Kind:** driver

## Purpose

`SimIssEphemeris` propagates a circular Keplerian ISS orbit in ECI meters from TLE
mean elements in `EphemerisConfig`. It satisfies `IssEphemeris` structurally.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SimIssEphemeris` | class | Circular ECI ephemeris stand-in |

## Inputs and outputs

Construction takes a `Clock` and optional `EphemerisConfig`.
`read_state(now_utc_s)` returns `Ok(IssState)` with position and velocity in meters.

## Behavior

1. Mean motion and inclination come from config. Semi-major axis follows
   `a = (mu / n^2)^(1/3)`.
2. At epoch the satellite is at the ascending node.
3. Position and velocity are evaluated on the circular orbit at `now_utc_s`.

## Errors and faults

None. The sim driver always returns `Ok`.

## Messages

None.

## Configuration

Reads `EphemerisConfig` mean motion, inclination, `mu`, and epoch.

## Constraints

There is no SGP4. The station TLE API is out of scope for this driver.

## Related documents

- [`flight.hal.interfaces.ephemeris`](../interfaces/ephemeris.md)
- [`flight.hal.drivers_sim`](../drivers_sim.md)
- [`flight.hal.drivers_real.ephemeris`](../drivers_real/ephemeris.md)
