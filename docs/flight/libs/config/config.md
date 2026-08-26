# flight.libs.config.config

**Source:** `packages/flight/src/flight/libs/config/config.py`
**Kind:** pure module

## Purpose

The module defines frozen dataclasses for all tunable flight parameters. Default field values
match `config/default.toml`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ControllerConfig` | class | Gimbal controller, tracker, LQR, and runaway tuning |
| `InferenceConfig` | class | Model paths, input bands, and latency budget |
| `CommsConfig` | class | Downlink/uplink rates, APID, and pass budgets |
| `StorageConfig` | class | Data root, capacity, and checksum algorithm |
| `SensorConfig` | class | Sensor geometry, mosaic layout, and calibration dir |
| `PreprocessingConfig` | class | Quality-flag thresholds |
| `FaultConfig` | class | Watchdog, inference timeout, thermal and power limits |
| `GimbalConfig` | class | Travel limits, stow/home poses, sim dynamics, serial link |
| `LinkConfig` | class | TCP/UDP endpoints and CCSDS APIDs |
| `CommandIngressConfig` | class | HMAC key path, auth flag, accepted sources |
| `CommandRouterConfig` | class | Hazardous ARM window duration |
| `EnvironmentConfig` | class | Per-axis sim/real wiring selector |
| `PactConfig` | class | Top-level config composing all sub-configs |
| `AxisMode` | type alias | `"sim"` or `"real"` |

## Inputs and outputs

Each config class is constructed with keyword arguments or defaults. `PactConfig()` with no
arguments yields a fully functional development configuration.

`config_loader.load_config()` is the sole TOML entry point. It returns `PactConfig`.

## Behavior

1. Each subsystem receives its sub-config slice at construction time.
2. Frozen dataclasses prevent runtime mutation after load.
3. Tuple fields hold array-like values. TOML arrays load as lists and map into tuples.
4. `EnvironmentConfig` names sim/real axes for sensor, gimbal, compute, link, and clock.
5. `LinkConfig` holds TCP bind for inbound TC and UDP destination for outbound TM.
6. `CommandIngressConfig` names the HMAC key path and accepted command sources.
7. Routable targets and hazardous commands come from the command dictionary, not from router
   config fields.

## Errors and faults

None from this module. Invalid TOML or out-of-range values fail in `config_loader` at startup.

## Messages

None.

## Configuration

The module defines configuration. Key field groups:

### ControllerConfig

Confidence gate, EMA alpha, deadband limits, retarget rate, slew limits, persistence frame
counts, scan timing, blob IoU threshold, Kalman noise parameters, LQR cost weights, and encoder
runaway tolerance.

### InferenceConfig

`model_path` (active segmentor), `classifier_model_path`, `rollback_model_path`,
`classifier_rollback_model_path`, `classifier_logit_threshold`, `input_bands`, input
dimensions, INT8 flag, and `latency_budget_ms`.

### CommsConfig

Downlink and uplink rate caps, daily byte caps, comm window weekdays, CCSDS APID, staged model
path, and per-pass downlink byte budget.

### SensorConfig

`width_px`, `height_px`, `bit_depth`, `mosaic_layout`, `ifov_deg_per_px`, default exposure
and gain, and `calibration_dir`.

### FaultConfig

`watchdog_interval_s`, `watchdog_max_miss_count`, `inference_timeout_ms`, `thermal_limit_c`,
and `power_limit_w`.

### GimbalConfig

Azimuth and elevation travel limits, stow and home poses, sim dynamics parameters, and PTU
serial settings.

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

## Related documents

- [`flight.libs.config`](../config.md)
- [`flight.libs.commands.dictionary`](../commands/dictionary.md)
