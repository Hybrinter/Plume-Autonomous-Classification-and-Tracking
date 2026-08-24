# flight.hal.interfaces.scalar

**Source:** `packages/flight/src/flight/hal/interfaces/scalar.py`
**Kind:** module

## Purpose

Defines the housekeeping scalar sensor Protocol. A driver returns one float reading per
call. The thermal and electrical subsystems share this interface. Each subsystem owns
the meaning and units of its reading.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ScalarSensor` | class | Runtime-checkable Protocol for a single-value sensor |

## Inputs and outputs

| Method | Inputs | Output |
| --- | --- | --- |
| `read` | none | `Result[float, FaultCode]` |

## Behavior

1. `read` samples the current scalar reading and returns it as a float.

## Errors and faults

`read` may return `Err(code)` on a read error. The Protocol does not define a fixed
fault code. The real stub driver never returns `Err`.

## Messages

None.

## Configuration

None.

## Constraints

- One float value per call.
- Units and semantics are defined by the consuming subsystem, not the driver.

## Related documents

- [`flight.hal.interfaces`](../interfaces.md)
- [`flight.hal.drivers_real.scalar`](../drivers_real/scalar.md)
- [`flight.hal.drivers_sim.scalar`](../drivers_sim/scalar.md)
