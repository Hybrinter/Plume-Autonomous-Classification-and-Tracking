# flight.hal.drivers_real.sensor

**Source:** `packages/flight/src/flight/hal/drivers_real/sensor.py`
**Kind:** driver

## Purpose

Captures raw mosaic frames from a FLIR Blackfly S camera through PySpin. The driver
satisfies `ImagingSensor` structurally. It imports PySpin inside `__init__`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `RealSensor` | class | PySpin camera driver |

## Inputs and outputs

Constructor:

- `clock` (`Clock`): wall-clock timestamps for each frame
- `serial_number` (`str | None`): camera serial; `None` selects the first enumerated
  device
- `timeout_ms` (int): `GetNextImage` timeout, default 1000 ms

Raises `ImportError` when PySpin is absent.

Protocol methods match `ImagingSensor`. See
[`flight.hal.interfaces.sensor`](../interfaces/sensor.md).

## Behavior

1. The constructor obtains the PySpin system singleton, selects a camera, and calls
   `Init()`. Acquisition does not start until `start_acquisition`.
2. `acquire_frame` calls `GetNextImage` under a lock.
3. An incomplete image or a Spinnaker exception returns `Err(CAMERA_STALL)`.
4. On success the driver copies the raw uint16 mosaic, releases the SDK buffer, and
   increments an internal `frame_id` counter starting at 1.
5. The returned `MosaicFrame` carries `timestamp_utc` from `clock.wall_clock_iso()`,
   current exposure and gain from the node map, and the mosaic array.
6. `set_exposure_us` and `set_gain_db` write the ExposureTime and Gain nodes under the
   same lock.
7. `start_acquisition` and `stop_acquisition` call `BeginAcquisition` and
   `EndAcquisition` under the lock.
8. A threading lock serializes capture and control-plane access.

## Errors and faults

| Fault | Trigger |
| --- | --- |
| `FaultCode.CAMERA_STALL` | SDK timeout, Spinnaker exception, or incomplete image transfer |

All control-plane methods map SDK errors to `CAMERA_STALL`.

## Messages

None.

## Configuration

Startup exposure and gain are applied by `select_drivers` from `PactConfig.sensor`
after construction.

## Constraints

- The driver acquires only. It does not demosaic or normalize.
- `frame_id` is a driver-assigned uint32 counter.
- The PySpin system and camera handle live for the driver lifetime.

## Related documents

- [`flight.hal.drivers_real`](../drivers_real.md)
- [`flight.hal.interfaces.sensor`](../interfaces/sensor.md)
