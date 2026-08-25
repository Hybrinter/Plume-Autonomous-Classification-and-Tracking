# flight.hal.drivers_sim.scalar

**Source:** `packages/flight/src/flight/hal/drivers_sim/scalar.py`
**Kind:** driver

## Purpose

`SimScalarSensor` replays a fixed list of float readings in order. After the list ends, it
holds the final value. It satisfies `ScalarSensor` structurally for SIL and tests.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SimScalarSensor` | class | Scripted scalar reading replay driver |

## Inputs and outputs

Construction takes a non-empty `list[float]` of readings.

| Method | Inputs | Outputs |
| --- | --- | --- |
| `read()` | None | `Ok(float)` |

## Behavior

1. Each `read()` call returns the next value from the scripted list.
2. After the last value, further calls repeat the final reading.
3. The index advances on every call even after hold-last begins.

## Errors and faults

None under normal operation.

## Messages

None.

## Configuration

None. Readings are supplied at construction (`thermal_readings` or `power_readings` from
`SimDriverInputs`).

## Constraints

- End-of-script behavior holds the last reading forever, matching a live housekeeping sensor.
- The constructor requires a non-empty readings list.

## Related documents

- [`flight.hal.interfaces.scalar`](interfaces/scalar.md)
- [`flight.hal.drivers_sim`](drivers_sim.md)
- [`flight.hal.drivers_real.scalar`](drivers_real/scalar.md)
