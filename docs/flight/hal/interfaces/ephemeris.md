# flight.hal.interfaces.ephemeris

**Source:** `packages/flight/src/flight/hal/interfaces/ephemeris.py`
**Kind:** module

## Purpose

This module defines the ISS ephemeris Protocol. `IssEphemeris.read_state` returns
inertial ISS position and velocity in ECI meters.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `IssState` | dataclass | ECI position, velocity, UTC epoch, and frame tag |
| `IssEphemeris` | Protocol | Injected source of ISS ECI state |

## Inputs and outputs

`read_state(now_utc_s)` returns `Result[IssState, FaultCode]`. `IssState.r_m` and
`v_m_s` are meters and meters per second. `frame` is the tag `ECI`.

## Behavior

1. The outer loop calls `read_state` with `Clock.utc_s()`.
2. A sim driver returns a circular Keplerian state.
3. A real stub returns `Err`. The outer loop then uses `omega_t_nom = 0`.

## Errors and faults

The real stub returns `EPHEMERIS_FAULT`. The Protocol does not fix other fault
values.

## Messages

None.

## Configuration

None at the Protocol level. The sim driver reads `EphemerisConfig`.

## Constraints

Apps depend on this Protocol only. They do not import concrete drivers.

## Related documents

- [`flight.hal.interfaces`](../interfaces.md)
- [`flight.hal.drivers_sim.ephemeris`](../drivers_sim/ephemeris.md)
- [`flight.hal.drivers_real.ephemeris`](../drivers_real/ephemeris.md)
