# flight.core.select_drivers

**Source:** `packages/flight/src/flight/core/select_drivers.py`
**Kind:** module

## Purpose

The driver selector maps each `environment` axis in `PactConfig` to a sim stand-in or a real
HAL driver. It returns a `Drivers` bundle for `build_apps`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SimDriverInputs` | class | Sim-only construction inputs for replay drivers |
| `select_drivers` | function | Resolve environment axes to a `Drivers` bundle |

## Inputs and outputs

**`SimDriverInputs` fields:**

- `frames`: mosaic frames for `SimSensor`
- `detector`: scripted detector for the compute axis
- `inbound_packets`: CCSDS TC packets for `SimStationLink`
- `thermal_readings`: Celsius readings for the thermal scalar sensor
- `power_readings`: watt readings for the electrical scalar sensor
- `launch_lock_engaged`: initial launch-lock state (default false)

**`select_drivers(config, clock, sim_inputs=None) -> Drivers`**

- Inputs: `PactConfig`, injected `Clock`, optional `SimDriverInputs`.
- Output: `Drivers` with each axis resolved.
- Raises `ValueError` when any axis is `sim` and `sim_inputs` is `None`.
- Raises `SystemExit` when real-sensor exposure or gain setup returns `Err`.
- Raises `ValueError` when the real compute axis loads an ONNX file whose
  shapes do not match the inference I/O contract.

## Behavior

1. Read `config.environment` axis values.
2. **Sensor axis:** `sim` selects `SimSensor` and `SimScalarSensor` pairs for thermal and
   power. `real` selects `RealSensor`, applies initial exposure and gain, and selects
   `RealScalarSensor` for both scalars.
3. **Gimbal axis:** `sim` selects `SimGimbal`. `real` selects `RealGimbal`.
4. **Compute axis:** `sim` uses the passed `ScriptedDetector`. `real` constructs
   `OnnxDetector` from the paths returned by `resolve_quantized_path` on
   `inference.segmentor_model_path` and `inference.classifier_model_path`.
   `use_int8` true selects `<stem>.int8.onnx`. The logit threshold and
   `fault.inference_timeout_ms` (20 ms) feed the detect-time fault threshold.
   The I/O contract is `(1, C, H, W)` from `input_bands` and `input_*_px`.
5. **Link axis:** `sim` selects `SimStationLink`. `real` selects `RealStationLink`.
6. **Launch lock:** always `SimLaunchLock`. Flight with `sim_inputs=None` starts ENGAGED.
   SIL uses `sim_inputs.launch_lock_engaged` (default RELEASED).
7. Return the assembled `Drivers` dataclass.

Real driver SDK modules import lazily inside the `real` branches only.

## Errors and faults

- `ValueError`: a `sim` axis without `sim_inputs`, or a real ONNX artifact
  whose shapes do not match the inference I/O contract.
- `SystemExit`: real-sensor exposure or gain command failure.

## Messages

None.

## Configuration

Reads `PactConfig.environment` axes (`sensor`, `gimbal`, `compute`, `link`) and per-driver
sub-configs (`sensor`, `gimbal`, `inference`, `fault`, `link`). The clock axis is handled by
the caller before this function runs. Real compute uses `resolve_quantized_path` when
`inference.use_int8` is true.

## Constraints

- This module imports both `drivers_sim` and lazy `drivers_real` branches.
- SDK modules (PySpin, pyserial, onnxruntime, socket) load only inside `real` branches.
- No real launch-lock driver exists. Every profile uses `SimLaunchLock`.
- Each branch local is typed with its HAL protocol. No casts are used at construction.

## Related documents

- [`flight.core`](../core.md)
- [`flight.core.composition`](composition.md)
- [`flight.core.main`](main.md)
- [`flight.payload.inference.artifact_path`](../payload/inference/artifact_path.md)
