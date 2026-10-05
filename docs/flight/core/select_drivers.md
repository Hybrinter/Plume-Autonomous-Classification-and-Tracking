# flight.core.select_drivers

**Source:** `packages/flight/src/flight/core/select_drivers.py`
**Kind:** module

## Purpose

The driver selector maps each `drivers` axis in `PactConfig` to a sim stand-in or a real
HAL driver. It returns a `Drivers` bundle for `build_apps`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SimDriverInputs` | class | Sim-only construction inputs for replay drivers |
| `select_drivers` | function | Resolve driver axes to a `Drivers` bundle |

## Inputs and outputs

**`SimDriverInputs` fields:**

- `frames`: mosaic frames for `SimSensor`
- `detector`: scripted detector for the compute axis
- `inbound_packets`: CCSDS TC packets for `SimStationLink`
- `thermal_readings`: Celsius readings for the thermal scalar sensor
- `power_readings`: watt readings for the electrical scalar sensor

**`select_drivers(config, clock, sim_inputs=None) -> Drivers`**

- Inputs: `PactConfig`, injected `Clock`, optional `SimDriverInputs`.
- Output: `Drivers` with each axis resolved.
- Raises `ValueError` when any axis is `sim` and `sim_inputs` is `None`.

## Behavior

1. Read `config.drivers` axis values.
2. **Sensor axis:** `sim` selects `SimSensor` and `SimScalarSensor` pairs for thermal and
   power. `real` selects `RealSensor` and `RealScalarSensor` for both scalars.
   No startup exposure or gain is applied here: the payload's graph-owned
   imaging policy applies settings through its stop/apply/start handling once
   a policy activates.
3. **Gimbal axis:** `sim` selects `SimGimbal`. `real` selects `RealGimbal`.
4. **Ephemeris axis:** `sim` selects `SimIssEphemeris`. `real` selects
   `RealIssEphemeris`.
5. **Compute axis:** `sim` returns `InferenceRuntime.from_scripted` over the
   passed `ScriptedDetector` - an explicit preinstalled scripted session.
   `real` returns an empty `InferenceRuntime` carrying the lazy
   `OnnxRuntimeFactory(config)`: no model file is read and no ONNX session is
   constructed here; the INIT lifecycle calls `load`, which resolves the
   quantized artifact paths, verifies digests and the I/O contract, and returns
   a typed `Err` on failure without raising.
6. **Link axis:** `sim` selects `SimStationLink`. `real` selects `RealStationLink`.
7. Return the assembled `Drivers` dataclass.

Real driver SDK modules import lazily inside the `real` branches only.

## Errors and faults

- `ValueError`: a `sim` axis without `sim_inputs`.

## Messages

None.

## Configuration

Reads `PactConfig.drivers` axes (`sensor`, `gimbal`, `ephemeris`, `compute`, `link`)
and per-driver sub-configs (`sensor`, `gimbal`, `ephemeris`, `inference`, `fault`,
`link`). The clock axis is handled by the caller before this function runs. Real compute
uses `resolve_quantized_path` when `inference.use_int8` is true.

## Constraints

- This module imports both `drivers_sim` and lazy `drivers_real` branches.
- SDK modules (PySpin, onnxruntime, socket) load only inside `real` branches.
- Each branch local is typed with its HAL protocol. No casts are used at construction.

## Related documents

- [`flight.core`](../core.md)
- [`flight.core.composition`](composition.md)
- [`flight.core.main`](main.md)
- [`flight.payload.inference.artifact_path`](../payload/inference/artifact_path.md)
