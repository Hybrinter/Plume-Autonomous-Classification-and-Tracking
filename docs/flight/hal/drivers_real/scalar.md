# flight.hal.drivers_real.scalar

**Source:** `packages/flight/src/flight/hal/drivers_real/scalar.py`
**Kind:** stub

## Purpose

`RealScalarSensor` is a placeholder scalar driver. It always returns `Ok(0.0)`. It satisfies
`ScalarSensor` structurally. Flight housekeeping bus integration is not wired yet.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `RealScalarSensor` | class | Stub scalar sensor (fixed 0.0 reading) |

## Inputs and outputs

| Method | Inputs | Outputs |
| --- | --- | --- |
| `read()` | None | `Ok(0.0)` always |

## Behavior

1. Every `read()` call returns `Ok(0.0)`.
2. No hardware access occurs.

## Errors and faults

None. The stub never returns `Err`.

## Messages

None.

## Configuration

None.

## Constraints

- This unit is a stub pending hardware integration.
- Tests and SIL use `SimScalarSensor` for scripted readings.
- The driver accepts no constructor arguments.

## Related documents

- [`flight.hal.interfaces.scalar`](interfaces/scalar.md)
- [`flight.hal.drivers_real`](drivers_real.md)
- [`flight.hal.drivers_sim.scalar`](drivers_sim/scalar.md)
