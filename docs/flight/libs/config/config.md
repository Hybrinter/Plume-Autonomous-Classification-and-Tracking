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
| `ArbiterConfig` | class | TRACKING / REWIND / SAFE persistence and limb arrival |
| `VisionConfig` | class | Blob gates and in-process vision queue depth |
| `InnerLoopConfig` | class | Inner PI, computed-torque, and encoder-rate fit |
| `OuterLoopConfig` | class | Outer period, Kp, and REWIND sharp-window duration |
| `PredictorConfig` | class | CoG/boresight intersect tracking proxy height |
| `ResidualConfig` | class | Residual KF noise, P0, and rewind ring |
| `PositionLoopConfig` | class | STOW / HOME / GOTO rate into the inner PI |
| `IntegrityConfig` | class | Catch-up cap and light GIMBAL_RUNAWAY detector |
| `ControllerConfig` | class | Nested vision, arbiter, inner, outer, residual, position, integrity, and predictor configs |
| `InferenceConfig` | class | Model paths, input bands, tensor size, and latency budget |
| `CommsConfig` | class | Downlink/uplink rates, APID, and pass budgets |
| `StorageConfig` | class | Data root, capacity, and checksum algorithm |
| `SensorConfig` | class | AP-3200T-USB frame, channel layout, optics, and capture limits |
| `PreprocessingConfig` | class | Quality-flag thresholds |
| `FaultConfig` | class | Watchdog, inference timeout, and power limit |
| `ThermalConfig` | class | Record-only per-component temperature limits |
| `GimbalConfig` | class | Elevation envelopes, stow/home, plant scalars, encoder |
| `LinkConfig` | class | TCP/UDP endpoints and CCSDS APIDs |
| `CommandIngressConfig` | class | HMAC key path, auth flag, accepted sources |
| `CommandRouterConfig` | class | Hazardous ARM window duration |
| `EphemerisConfig` | class | Circular-orbit ISS elements and WGS-84 constants |
| `DriverConfig` | class | Per-axis sim/real wiring selector |
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
5. `PactConfig` requires inference `H,W` to equal the sensor frame
   (`height_px`, `width_px`) and `input_bands` to be a subset of `channel_layout`.
6. `DriverConfig` names sim/real axes for sensor, gimbal, ephemeris, compute, link, and clock.
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

Nested tables under `[controller]`:

- `vision`: `confidence_gate`, `blob_iou_match_threshold`, `min_blob_area_px`,
  `queue_depth`
- `arbiter`: `release_persistence_frames`, `max_observation_age_s`, `limb_arrival_deg`
- `inner`: `dt_s`, `rate_fit_n`, `rate_fit_degree`, `kp`, `ki`, `tau_cl_s`
- `outer`: `dt_s`, `Kp`, `rewind_sharp_max_s`
- `predictor`: `cog_height_m`
- `residual`: `Q_diag`, `R_v`, `P0_diag`, `rewind_horizon_s`, `rewind_snapshots`
- `position`: `K_pos`, `r_max_deg_per_s`
- `integrity`: `catchup_max_s`, `freeze_strikes`, `r_min_rad_s`,
  `encoder_rate_ratio`, `command_authority_s`, `feedback_max_age_s`, `recovery_max_attempts`,
  `recovery_window_s`, `science_boundary_guard_deg`

### InferenceConfig

`segmentor_model_path`, `classifier_model_path`, `segmentor_rollback_model_path`,
`classifier_rollback_model_path`, `classifier_logit_threshold`, `input_bands`, input
dimensions (`1544 x 2064`), INT8 flag, and `latency_budget_ms` (4 ms expected
detect). `input_bands` is BLUE, GREEN, RED.

### CommsConfig

Downlink and uplink rate caps, daily byte caps, comm window weekdays, CCSDS APID,
staged segmentor and classifier paths, and per-pass downlink byte budget.

### SensorConfig

`width_px` (lateral 2064) and `height_px` (along-track 1544), bit depth, wire
`channel_layout` RED/GREEN/BLUE, and pixel pitch. `SensorOpticsConfig` holds the
Edmund 16-849 focal length, f-number, stored distortion, one-pixel IFOV, active-area
FOV, and the 1/1.8 in datasheet HFOV. `SensorCaptureConfig` holds the 35 Hz cap,
8-bit exposure range, ALC gain range, and `duty_cycle`. `calibration_dir` selects
artifact loading. Duty 0.5 is the imaging gate, not the Xeryon vacuum duty.

### FaultConfig

`watchdog_interval_s`, `watchdog_max_miss_count`, `inference_timeout_ms` (20 ms),
and `power_limit_w` (payload-bus FDIR; module Super TDP is 25 W).

### ThermalConfig

Per-component min/max Celsius records: camera, lens, gimbal, compute. Housekeeping does
not compare these values.

### GimbalConfig

Hardware elevation `[0, +90]` deg, science window `[+5, +45]`, stow at `+90`
(flat launch pose), home at `+45`, and max hardware slew. Plant `J_kg_m2` is capped
by the XRT-U-60 payload inertia limit, with `B_nms_per_rad`, `tau_max_nm` (90 mN·m),
64800 encoder counts, and sim encoder noise. There is no azimuth travel field.

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
- Science elevation must lie inside hardware travel. Stow and home must lie inside
  hardware travel. Plant `J_kg_m2` must be at most `xeryon.payload_inertia_limit_kg_m2`.
- `rate_fit_n` must be greater than `rate_fit_degree`. `Q_diag` and `P0_diag` have
  length 2.

## Related documents

- [`flight.libs.config`](../config.md)
- [`flight.libs.commands.dictionary`](../commands/dictionary.md)
