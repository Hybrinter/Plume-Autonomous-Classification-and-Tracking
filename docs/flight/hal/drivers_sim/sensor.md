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
| `load_next(frame)` | `MosaicFrame` | None (sim-only slot write) |
| `unread_scripted_count()` | None | Remaining constructor frames |
| `set_exposure_us(exposure)` | Microseconds (ignored) | `Ok(None)` |
| `set_gain_db(gain)` | dB (ignored) | `Ok(None)` |
| `start_acquisition()` | None | `Ok(None)` |
| `stop_acquisition()` | None | `Ok(None)` |

## Behavior

1. Each `acquire_frame()` call returns the live slot when one is set.
2. Otherwise it returns the next frame from the constructor list.
3. After both the slot and the list are empty, `acquire_frame()` returns
   `Err(CAMERA_STALL)`.
4. `load_next` overwrites the unread live slot. It is not on `ImagingSensor`.
5. `unread_scripted_count` returns constructor frames that `acquire_frame` has
   not yet returned. The live slot does not change this count.
6. Exposure, gain, and acquisition control calls are no-ops that always succeed.
7. `start_acquisition()` and `stop_acquisition()` toggle an internal acquiring flag only.

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
- `load_next` is a sim-only mutator. Real drivers have no counterpart.
- Callers must not overlap `load_next` with `acquire_frame`. The slot has no lock.
- `unread_scripted_count` is a sim-only observer. Real drivers have no counterpart.

## Related documents

- [`flight.hal.interfaces.sensor`](interfaces/sensor.md)
- [`flight.hal.drivers_sim`](drivers_sim.md)
- [`flight.hal.drivers_real.sensor`](drivers_real/sensor.md)
