# flight.hal.interfaces.scalar

**Source:** `packages/flight/src/flight/hal/interfaces/scalar.py`
**Kind:** module

## Purpose

This module defines the `ScalarSensor` Protocol for a single float housekeeping reading.
Thermal and electrical apps share this surface. Each app owns the meaning and units of its
reading.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ScalarSensor` | Protocol | Single-value sensor read surface |

## Inputs and outputs

| Method | Inputs | Outputs |
| --- | --- | --- |
| `read()` | None | `Result[float, FaultCode]` |

## Behavior

1. The consuming app calls `read()` on its polling interval.
2. A successful call returns one scalar sample.
3. The app interprets the float (temperature in Celsius, power in Watts, and so on).

## Errors and faults

Implementations return `Err(FaultCode)` on a read failure. The Protocol does not fix the
fault code.

## Messages

None.

## Configuration

None at the Protocol level. The composition root injects separate sensor instances for
thermal and electrical subsystems.

## Constraints

- One float per read. No batch or vector surface on this Protocol.
- Units and physical meaning are owned by the consuming subsystem, not the driver.

## Related documents

- [`flight.hal.interfaces`](interfaces.md)
- [`flight.hal.drivers_real.scalar`](drivers_real/scalar.md)
- [`flight.hal.drivers_sim.scalar`](drivers_sim/scalar.md)
- [`flight.thermal`](thermal.md)
- [`flight.electrical`](electrical.md)
