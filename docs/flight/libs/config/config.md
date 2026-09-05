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
| `ControllerConfig` | class | Cascaded elevation controller, tracker, and residual KF |
| `InferenceConfig` | class | Model paths, input bands, tensor size, and latency budget |
| `CommsConfig` | class | Downlink/uplink rates, APID, and pass budgets |
| `StorageConfig` | class | Data root, capacity, and checksum algorithm |
| `SensorConfig` | class | Purchased mosaic geometry, optics, IFOV, and exposure/gain ranges |
| `PreprocessingConfig` | class | Quality-flag thresholds |
| `FaultConfig` | class | Watchdog, inference timeout, and power limit |
| `ThermalConfig` | class | Record-only per-component temperature limits |
| `GimbalConfig` | class | Elevation envelopes, stow/home, plant scalars, encoder |
| `LinkConfig` | class | TCP/UDP endpoints and CCSDS APIDs |
| `CommandIngressConfig` | class | HMAC key path, auth flag, accepted sources |
| `CommandRouterConfig` | class | Hazardous ARM window duration |
| `EphemerisConfig` | class | Circular-orbit ISS elements and WGS-84 constants |
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
6. `EnvironmentConfig` names sim/real axes for sensor, gimbal, ephemeris, compute, link, and clock.
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

Confidence and area gates, persistence frames, blob IoU, inner PI and SG window, outer
`Kp` and residual KF (`Q_diag`, `R_v`, `P0_diag`), rewind snapshots, vision queue depth,
and STOW/HOME position-loop gains.

### InferenceConfig

`segmentor_model_path`, `classifier_model_path`, `segmentor_rollback_model_path`,
`classifier_rollback_model_path`, `classifier_logit_threshold`, `input_bands`, input
dimensions (`1024 x 1224`), INT8 flag, and `latency_budget_ms` (4 ms expected
detect).

### CommsConfig

Downlink and uplink rate caps, daily byte caps, comm window weekdays, CCSDS APID,
staged segmentor and classifier paths, and per-pass downlink byte budget.

### SensorConfig

Mosaic `width_px` (lateral 2448) and `height_px` (along-track 2048), bit depth, mosaic
layout, pixel pitch, focal length, f-number, mosaic and band IFOV, FOV check fields,
QE and well-capacity records, exposure and gain legal ranges plus initials, and
`calibration_dir`.

### FaultConfig

`watchdog_interval_s`, `watchdog_max_miss_count`, `inference_timeout_ms` (20 ms),
and `power_limit_w` (payload-bus FDIR; module Super TDP is 25 W).

### ThermalConfig

Per-component min/max Celsius records: camera, lens, gimbal, compute. Housekeeping does
not compare these values.

### GimbalConfig

Hardware elevation `[el_hw_min_deg, el_hw_max_deg]`, science window
`[el_science_min_deg, el_science_max_deg]`, stow and home elevation, max hardware slew,
plant copies `J_kg_m2`, `B_nms_per_rad`, `tau_max_nm`, 18-bit encoder counts, and sim
encoder noise. There is no azimuth travel field.

### LinkConfig

`command_tcp_host`, `command_tcp_port`, `telemetry_udp_host`, `telemetry_udp_port`,
`socket_timeout_s`, `tc_apid`, and `tm_apid`.

### CommandIngressConfig

`hmac_key_path`, `require_auth`, and `accepted_sources`.

### CommandRouterConfig

`arm_window_s` for hazardous ARM/EXECUTE pairing.

### EphemerisConfig

ISS circular-orbit mean elements (`inclination_deg`, `mean_motion_rev_per_day`,
`mu_m3_s2`, `epoch_utc_s`), Earth rate, and WGS-84 `a` and `f`.

## Constraints

- Default field values must match `config/default.toml` exactly.
- No subsystem reads TOML directly.
- `calibration_dir=""` selects identity calibration (SIL only).
- Launch-lock axis is not in `EnvironmentConfig`.
- Science elevation must lie inside hardware travel. Stow and home must lie inside
  hardware travel.
- `rate_fit_n` must be greater than `rate_fit_degree`. `Q_diag` and `P0_diag` have
  length 2.

## Related documents

- [`flight.libs.config`](../config.md)
- [`flight.libs.commands.dictionary`](../commands/dictionary.md)
