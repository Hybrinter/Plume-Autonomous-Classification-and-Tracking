# flight.libs.config.config

**Source:** `packages/flight/src/flight/libs/config/config.py`
**Kind:** pure module

## Purpose

The module defines frozen schema dataclasses for all tunable flight parameters. Default field
values match `config/default.toml`. Field constraints and cross-field checks run when a config
object is constructed.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ControllerConfig` | class | Gimbal controller, tracker, LQR, and runaway tuning |
| `InferenceConfig` | class | Model paths, input bands, tensor size, and latency budget |
| `CommsConfig` | class | Downlink/uplink rates, APID, and pass budgets |
| `StorageConfig` | class | Data root, capacity, and checksum algorithm |
| `SensorConfig` | class | Purchased mosaic geometry, optics, IFOV, and exposure/gain ranges |
| `PreprocessingConfig` | class | Quality-flag thresholds |
| `FaultConfig` | class | Watchdog, inference timeout, and power limit |
| `ThermalConfig` | class | Record-only per-component temperature limits |
| `GimbalConfig` | class | Hardware and science elevation envelopes, stow/home, sim, serial |
| `LinkConfig` | class | TCP/UDP endpoints and CCSDS APIDs |
| `CommandIngressConfig` | class | HMAC key path, auth flag, accepted sources |
| `CommandRouterConfig` | class | Hazardous ARM window duration |
| `EnvironmentConfig` | class | Per-axis sim/real wiring selector |
| `PactConfig` | class | Top-level config composing all sub-configs |
| `AxisMode` | type alias | `"sim"` or `"real"` |

## Inputs and outputs

Each config class is constructed with keyword arguments or defaults. `PactConfig()` with no
arguments yields a fully functional development configuration.

`config_loader.load_config()` is the sole TOML entry point. It validates a merged TOML dict
into `PactConfig`.

## Behavior

1. Each subsystem receives its sub-config slice at construction time.
2. Frozen dataclasses prevent runtime mutation after load.
3. Tuple fields hold array-like values. TOML arrays load as lists and map into tuples.
4. Unknown keys and out-of-range values fail at construction.
5. `PactConfig` requires inference `H,W` to equal the demosaiced band plane
   (`height_px/2`, `width_px/2`).
6. `EnvironmentConfig` names sim/real axes for sensor, gimbal, compute, link, and clock.
7. `LinkConfig` holds TCP bind for inbound TC and UDP destination for outbound TM.
8. `CommandIngressConfig` names the HMAC key path and accepted command sources.
9. Routable targets and hazardous commands come from the command dictionary, not from router
   config fields.

## Errors and faults

Construction raises `ValidationError` for unknown keys, out-of-range fields, and cross-field
violations. `config_loader.load_config()` maps those errors to `Err(str)`.

## Messages

None.

## Configuration

The module defines configuration. Key field groups:

### ControllerConfig

Confidence gate, EMA alpha, retarget rate, persistence frame counts, blob IoU threshold,
Kalman noise parameters, LQR cost weights, and encoder runaway tolerance.

### InferenceConfig

`segmentor_model_path`, `classifier_model_path`, `segmentor_rollback_model_path`,
`classifier_rollback_model_path`, `classifier_logit_threshold`, `input_bands`, input
dimensions (`1024 x 1224`), INT8 flag, and `latency_budget_ms`.

### CommsConfig

Downlink and uplink rate caps, daily byte caps, comm window weekdays, CCSDS APID,
staged segmentor and classifier paths, and per-pass downlink byte budget.

### SensorConfig

Mosaic `width_px` (lateral 2448) and `height_px` (along-track 2048), bit depth, mosaic
layout, pixel pitch, focal length, f-number, mosaic and band IFOV, FOV check fields,
QE and well-capacity records, exposure and gain legal ranges plus initials, and
`calibration_dir`.

### FaultConfig

`watchdog_interval_s`, `watchdog_max_miss_count`, `inference_timeout_ms`, and
`power_limit_w`.

### ThermalConfig

Per-component min/max Celsius records: camera, lens, gimbal, compute. Housekeeping does
not compare these values.

### GimbalConfig

Hardware elevation `[el_hw_min_deg, el_hw_max_deg]`, science window
`[el_science_min_deg, el_science_max_deg]`, stow and home elevation, max hardware slew,
sim dynamics parameters, and PTU serial settings. Azimuth travel is not configured;
drivers pin azimuth at 0.

### LinkConfig

`command_tcp_host`, `command_tcp_port`, `telemetry_udp_host`, `telemetry_udp_port`,
`socket_timeout_s`, `tc_apid`, and `tm_apid`.

### CommandIngressConfig

`hmac_key_path`, `require_auth`, and `accepted_sources`.

### CommandRouterConfig

`arm_window_s` for hazardous ARM/EXECUTE pairing.

## Constraints

- Default field values must match `config/default.toml` exactly.
- No subsystem reads TOML directly.
- `calibration_dir=""` selects identity calibration (SIL only).
- `serial_port=""` marks real gimbal unavailable at startup.
- Launch-lock axis is not in `EnvironmentConfig`.
- Science elevation must lie inside hardware travel. Stow and home must lie inside
  hardware travel.

## Related documents

- [`flight.libs.config`](../config.md)
- [`flight.libs.commands.dictionary`](../commands/dictionary.md)
