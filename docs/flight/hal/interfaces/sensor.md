# flight.hal.interfaces.sensor

**Source:** `packages/flight/src/flight/hal/interfaces/sensor.py`
**Kind:** module

## Purpose

This module defines the `ImagingSensor` Protocol for a 2x2-CFA mosaic camera. Drivers
acquire raw frames only. Demosaic, calibration, and normalization run in
`flight.payload.preprocess`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ImagingSensor` | Protocol | Acquire-only imaging sensor surface |

## Inputs and outputs

| Method | Inputs | Outputs |
| --- | --- | --- |
| `acquire_frame()` | None | `Result[MosaicFrame, FaultCode]` |
| `set_exposure_us(exposure)` | Exposure in microseconds | `Result[None, FaultCode]` |
| `set_gain_db(gain)` | Analogue gain in dB | `Result[None, FaultCode]` |
| `start_acquisition()` | None | `Result[None, FaultCode]` |
| `stop_acquisition()` | None | `Result[None, FaultCode]` |

`MosaicFrame` carries a raw `(H, W)` uint16 mosaic plane plus capture metadata. It is not
a bus message.

## Behavior

1. The payload app calls `acquire_frame()` on the capture path.
2. A successful call returns a raw mosaic plane with no in-driver processing.
3. Control-plane calls adjust exposure, gain, and acquisition state.
4. Implementations serialize capture and control access when both paths are active.

## Errors and faults

| Fault | Trigger |
| --- | --- |
| `CAMERA_STALL` | No complete frame is available in time |

Control-plane methods may return other SDK-specific fault codes mapped by each driver.

## Messages

None.

## Configuration

None at the Protocol level. Concrete drivers read camera and timeout settings from
`PactConfig` at construction.

## Constraints

- Drivers acquire only. No demosaic, calibration, or normalization inside any driver.
- `MosaicFrame` passes by direct call from the sensor driver to the payload app.
- Implementations must be thread-safe between the capture loop and tuning calls.

## Related documents

- [`flight.hal.interfaces`](interfaces.md)
- [`flight.hal.drivers_real.sensor`](drivers_real/sensor.md)
- [`flight.hal.drivers_sim.sensor`](drivers_sim/sensor.md)
- [`flight.payload.preprocess`](payload/preprocess.md)
