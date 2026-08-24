# flight.hal.drivers_sim.sensor

**Source:** `packages/flight/src/flight/hal/drivers_sim/sensor.py`
**Kind:** driver

## Purpose

Replays a fixed list of raw `MosaicFrame` values in order. The driver satisfies
`ImagingSensor` structurally. It performs no image processing.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SimSensor` | class | Scripted mosaic frame replay driver |

## Inputs and outputs

Constructor:

- `frames` (`list[MosaicFrame]`): ordered frames to return one per `acquire_frame`
  call

Protocol methods match `ImagingSensor`. See
[`flight.hal.interfaces.sensor`](../interfaces/sensor.md).

## Behavior

1. `acquire_frame` returns the next frame from the list and advances an internal index.
2. When the list is exhausted, `acquire_frame` returns `Err(CAMERA_STALL)`.
3. `set_exposure_us` and `set_gain_db` are no-ops. They always return `Ok(None)`.
4. `start_acquisition` sets an internal acquiring flag to `True`.
5. `stop_acquisition` sets the acquiring flag to `False`.

## Errors and faults

| Fault | Trigger |
| --- | --- |
| `FaultCode.CAMERA_STALL` | The replay list is exhausted |

## Messages

None.

## Configuration

None. Frames are supplied at construction by the SIL or test harness.

## Constraints

- End-of-script behavior matches a stalled camera.
- The acquiring flag does not gate `acquire_frame`.
- Frames are typically rendered by `sim.scene` before injection.

## Related documents

- [`flight.hal.drivers_sim`](../drivers_sim.md)
- [`flight.hal.interfaces.sensor`](../interfaces/sensor.md)
