# flight.hal.drivers_sim.scalar

**Source:** `packages/flight/src/flight/hal/drivers_sim/scalar.py`
**Kind:** driver

## Purpose

Replays a fixed list of float readings in order. The driver satisfies `ScalarSensor`
structurally. It holds the final value once the list is exhausted.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SimScalarSensor` | class | Scripted scalar reading replay driver |

## Inputs and outputs

Constructor:

- `readings` (`list[float]`): non-empty ordered readings

| Method | Inputs | Output |
| --- | --- | --- |
| `read` | none | `Result[float, FaultCode]` |

## Behavior

1. `read` returns the reading at the current index and advances the index.
2. After the last reading is consumed, further calls repeat the final value.
3. Every call returns `Ok(value)`.

## Errors and faults

None. The driver never returns `Err`.

## Messages

None.

## Configuration

None. Readings are supplied at construction by the SIL or test harness.

## Constraints

- The constructor requires a non-empty `readings` list.
- End-of-script behavior holds the last reading forever.
- Two instances serve the thermal and power sensors in SIL.

## Related documents

- [`flight.hal.drivers_sim`](../drivers_sim.md)
- [`flight.hal.interfaces.scalar`](../interfaces/scalar.md)
- [`flight.hal.drivers_real.scalar`](../drivers_real/scalar.md)
