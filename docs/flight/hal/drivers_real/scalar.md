# flight.hal.drivers_real.scalar

**Source:** `packages/flight/src/flight/hal/drivers_real/scalar.py`
**Kind:** stub

## Purpose

This module is a placeholder for real housekeeping scalar sensors. It satisfies
`ScalarSensor` structurally. It always returns `Ok(0.0)`. Tests and CI use
`SimScalarSensor`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `RealScalarSensor` | class | Stub scalar sensor driver |

## Inputs and outputs

| Method | Inputs | Output |
| --- | --- | --- |
| `read` | none | `Result[float, FaultCode]` — always `Ok(0.0)` |

The constructor takes no arguments.

## Behavior

1. `read` returns `Ok(0.0)` on every call.

## Errors and faults

None. The stub never returns `Err`.

## Messages

None.

## Configuration

None.

## Constraints

- This module is a stub pending hardware integration.
- The nominal reading is always 0.0.
- Two instances are injected as the thermal and power sensors when the profile selects
  real scalar sensors.

## Related documents

- [`flight.hal.drivers_real`](../drivers_real.md)
- [`flight.hal.interfaces.scalar`](../interfaces/scalar.md)
- [`flight.hal.drivers_sim.scalar`](../drivers_sim/scalar.md)
