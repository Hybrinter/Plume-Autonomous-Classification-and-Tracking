# flight.hal.drivers_real.sensor

**Source:** `packages/flight/src/flight/hal/drivers_real/sensor.py`
**Kind:** driver

## Purpose

`RealSensor` drives a FLIR Blackfly S camera through PySpin. It captures one raw 2x2-CFA
mosaic frame per `acquire_frame()` call. It satisfies `ImagingSensor` structurally.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `RealSensor` | class | PySpin-backed imaging sensor driver |

## Inputs and outputs

Construction takes a `Clock`, an optional camera serial number, and a frame timeout in
milliseconds.

| Method | Inputs | Outputs |
| --- | --- | --- |
| `acquire_frame()` | None | `Result[MosaicFrame, FaultCode]` |
| `set_exposure_us(exposure)` | Microseconds | `Result[None, FaultCode]` |
| `set_gain_db(gain)` | dB | `Result[None, FaultCode]` |
| `start_acquisition()` | None | `Result[None, FaultCode]` |
| `stop_acquisition()` | None | `Result[None, FaultCode]` |

Construction raises `ImportError` when PySpin is not installed.

## Behavior

1. Construction opens the PySpin system, selects a camera by serial or index, and calls
   `Init()`.
2. `start_acquisition()` begins streaming. `acquire_frame()` waits up to the timeout for
   the next image.
3. A complete image copies into a uint16 numpy array and releases the SDK buffer.
4. The driver stamps `timestamp_utc` from the injected clock, sets `capture_monotonic_s`
   from `Clock.monotonic_s()`, and reads exposure and gain from the node map.
5. A lock serializes all node-map access between capture and control calls.
6. `stop_acquisition()` ends streaming.

## Errors and faults

| Fault | Trigger |
| --- | --- |
| `CAMERA_STALL` | SDK timeout, incomplete image, or node-map error |

## Messages

None.

## Configuration

| Field | Source | Use |
| --- | --- | --- |
| Camera serial | Constructor argument or first enumerated device | Device selection |
| `timeout_ms` | Constructor (default 1000) | `GetNextImage` timeout |

Startup exposure and gain come from `PactConfig` via `select_drivers`.

## Constraints

- PySpin imports inside `__init__` only. Importing this module does not require the SDK.
- Acquire-only: no demosaic, calibration, or normalization in the driver.
- `frame_id` is a driver-local uint32 counter starting at 1 on the first good frame.

## Related documents

- [`flight.hal.interfaces.sensor`](interfaces/sensor.md)
- [`flight.hal.drivers_real`](drivers_real.md)
- [`flight.hal.drivers_sim.sensor`](drivers_sim/sensor.md)
