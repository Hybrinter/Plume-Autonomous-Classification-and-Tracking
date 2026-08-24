# flight.hal.interfaces.sensor

**Source:** `packages/flight/src/flight/hal/interfaces/sensor.py`
**Kind:** module

## Purpose

Defines the imaging sensor Protocol. Drivers acquire raw mosaic frames only. They do
not demosaic, calibrate, or normalize. Those stages run in `flight.payload.preprocess`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ImagingSensor` | class | Runtime-checkable Protocol for a 2x2 mosaic camera |

## Inputs and outputs

| Method | Inputs | Output |
| --- | --- | --- |
| `acquire_frame` | none | `Result[MosaicFrame, FaultCode]` |
| `set_exposure_us` | `exposure` (float, microseconds) | `Result[None, FaultCode]` |
| `set_gain_db` | `gain` (float, dB) | `Result[None, FaultCode]` |
| `start_acquisition` | none | `Result[None, FaultCode]` |
| `stop_acquisition` | none | `Result[None, FaultCode]` |

`MosaicFrame` carries a raw `(H, W)` uint16 mosaic plane plus `timestamp_utc`,
`frame_id`, `exposure_us`, and `gain_db`.

## Behavior

1. `acquire_frame` captures one raw 2x2-CFA mosaic frame.
2. `set_exposure_us` writes the exposure time in microseconds.
3. `set_gain_db` writes the analogue gain in dB.
4. `start_acquisition` begins continuous frame acquisition.
5. `stop_acquisition` stops acquisition and releases buffers.

## Errors and faults

| Fault | Trigger |
| --- | --- |
| `FaultCode.CAMERA_STALL` | No complete frame is available in time |

Control-plane methods may return `Err` on an SDK or control error. The concrete driver
maps those to `FaultCode.CAMERA_STALL`.

## Messages

None. `MosaicFrame` passes by direct call from the injected sensor to the payload app.

## Configuration

None at the Protocol level. Real drivers read startup exposure and gain from
`PactConfig.sensor`.

## Constraints

- Implementations must be thread-safe. The capture path and control path may run on
  different threads.
- The returned plane is un-demosaicked raw CFA data.
- No driver performs image processing.

## Related documents

- [`flight.hal.interfaces`](../interfaces.md)
- [`flight.hal.drivers_real.sensor`](../drivers_real/sensor.md)
- [`flight.hal.drivers_sim.sensor`](../drivers_sim/sensor.md)
