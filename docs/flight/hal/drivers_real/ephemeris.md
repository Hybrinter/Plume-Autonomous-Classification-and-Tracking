# flight.hal.drivers_real.ephemeris

**Source:** `packages/flight/src/flight/hal/drivers_real/ephemeris.py`
**Kind:** driver

## Purpose

`RealIssEphemeris` is a stub. The station TLE API is not implemented.
`read_state` always returns `Err(EPHEMERIS_FAULT)`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `RealIssEphemeris` | class | Stub `IssEphemeris` |

## Inputs and outputs

`read_state(now_utc_s)` ignores the timestamp and returns `Err(EPHEMERIS_FAULT)`.

## Behavior

1. Construction opens no network socket and parses no TLE.
2. Every `read_state` call returns `Err(EPHEMERIS_FAULT)`.
3. The outer loop treats that as `omega_t_nom = 0`.

## Errors and faults

| Fault | Trigger |
| --- | --- |
| `EPHEMERIS_FAULT` | Every `read_state` call |

## Messages

None.

## Configuration

None.

## Constraints

This driver is a placeholder. It does not import a vendor SDK.

## Related documents

- [`flight.hal.interfaces.ephemeris`](../interfaces/ephemeris.md)
- [`flight.hal.drivers_real`](../drivers_real.md)
- [`flight.hal.drivers_sim.ephemeris`](../drivers_sim/ephemeris.md)
