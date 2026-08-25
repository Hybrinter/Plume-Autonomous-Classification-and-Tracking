# flight.hal.drivers_sim.sensor

**Source:** `packages/flight/src/flight/hal/drivers_sim/sensor.py`
**Kind:** driver

## Purpose

`SimSensor` replays a fixed list of `MosaicFrame` values in order. It satisfies
`ImagingSensor` structurally for SIL and tests.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SimSensor` | class | Scripted mosaic frame replay driver |

## Inputs and outputs

Construction takes an ordered `list[MosaicFrame]`.

| Method | Inputs | Outputs |
| --- | --- | --- |
| `acquire_frame()` | None | `Result[MosaicFrame, FaultCode]` |
| `set_exposure_us(exposure)` | Microseconds (ignored) | `Ok(None)` |
| `set_gain_db(gain)` | dB (ignored) | `Ok(None)` |
| `start_acquisition()` | None | `Ok(None)` |
| `stop_acquisition()` | None | `Ok(None)` |

## Behavior

1. Each `acquire_frame()` call returns the next frame from the scripted list.
2. After the list is exhausted, `acquire_frame()` returns `Err(CAMERA_STALL)`.
3. Exposure, gain, and acquisition control calls are no-ops that always succeed.
4. `start_acquisition()` and `stop_acquisition()` toggle an internal acquiring flag only.

## Errors and faults

| Fault | Trigger |
| --- | --- |
| `CAMERA_STALL` | The replay list is exhausted |

## Messages

None.

## Configuration

None. Frames are supplied at construction by the SIL or test harness.

## Constraints

- Acquire-only: the driver performs no image processing.
- End-of-script behavior matches a stalled camera, not a hold-last frame.
- Frames are typically rendered by `sim.scene`.

## Related documents

- [`flight.hal.interfaces.sensor`](interfaces/sensor.md)
- [`flight.hal.drivers_sim`](drivers_sim.md)
- [`flight.hal.drivers_real.sensor`](drivers_real/sensor.md)
