# flight.libs.config.config

**Source:** `packages/flight/src/flight/libs/config/config.py`
**Kind:** pure module

## Purpose

This module defines frozen dataclasses for all tunable flight parameters. `PactConfig`
composes every subsystem config. Default field values match `config/default.toml`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ControllerConfig` | dataclass | Gimbal controller, arbiter, Kalman, LQR, runaway tuning |
| `InferenceConfig` | dataclass | Model paths, input shape, latency budget |
| `CommsConfig` | dataclass | Downlink/uplink rates, daily caps, CCSDS APID, staging path |
| `StorageConfig` | dataclass | Data root, storage limit, checksum algorithm |
| `SensorConfig` | dataclass | Mosaic geometry, bit depth, IFOV, exposure, calibration dir |
| `PreprocessingConfig` | dataclass | Quality-flag thresholds |
| `FaultConfig` | dataclass | Watchdog, inference timeout, thermal and power limits |
| `GimbalConfig` | dataclass | Travel limits, stow/home poses, sim dynamics, serial link |
| `LinkConfig` | dataclass | TCP command bind, UDP telemetry endpoint, APIDs, socket timeout |
| `CommandIngressConfig` | dataclass | HMAC key path, auth flag, accepted command sources |
| `CommandRouterConfig` | dataclass | Hazardous-command ARM window duration |
| `EnvironmentConfig` | dataclass | Per-axis `sim`/`real` selectors and host label |
| `AxisMode` | type alias | Literal `"sim"` or `"real"` |
| `PactConfig` | dataclass | Top-level config composing all subsystem configs |

## Inputs and outputs

Constructors take field values only. `PactConfig()` with no arguments yields development
defaults. `config_loader` populates instances from merged TOML.

## Behavior

1. Each dataclass holds defaults for one subsystem or cross-cutting concern.
2. `PactConfig` nests one instance of each subsystem config via `default_factory`.
3. `EnvironmentConfig` names deployment axes: `sensor`, `gimbal`, `compute`, `link`,
   `clock`, and `host`. The composition root reads `clock` before building drivers.
4. `LinkConfig` sets TCP bind for inbound telecommands and UDP destination for outbound
   telemetry and products.
5. `CommandIngressConfig` names the HMAC key file path, `require_auth`, and
   `accepted_sources`. The composition root loads key bytes and injects them into
   `iss_iface`.
6. `CommandRouterConfig` sets `arm_window_s` for hazardous ARM/EXECUTE handling.
7. Routable targets and hazardous opcodes come from `flight.libs.commands`, not from this
   module.

### Default highlights

| Dataclass | Notable defaults |
| --- | --- |
| `ControllerConfig` | `confidence_gate=0.55`, `retarget_rate_limit_hz=0.5`, Kalman and LQR diagonals |
| `InferenceConfig` | `model_path="data/models/active.pt"`, `input_height_px=256`, `latency_budget_ms=500` |
| `CommsConfig` | `max_downlink_rate_bps=5_000_000`, `ccsds_apid=0x001` |
| `SensorConfig` | `1024x1024`, `bit_depth=12`, mosaic layout BLUE/GREEN/RED/NIR |
| `FaultConfig` | `watchdog_interval_s=5.0`, `watchdog_max_miss_count=3` |
| `GimbalConfig` | Az `-90..90`, El `-45..45`, `serial_baud=9600` |
| `LinkConfig` | TCP `50501`, UDP `50502`, `tc_apid=0x001`, `tm_apid=0x002` |

## Errors and faults

None. This module defines data only.

## Messages

None.

## Configuration

This module is the typed config schema. `config_loader` is the sole TOML entry point.

## Constraints

- All dataclasses use `@dataclass(frozen=True)`.
- Defaults must stay aligned with `config/default.toml` and `test_config_defaults`.
- Tuple defaults hold array-like values (for example `input_bands`, `lqr_Q_diag`).
- `EnvironmentConfig` has no launch-lock axis field.

## Related documents

- [`flight.libs.config`](flight/libs/config.md)
- [`flight.core.config_loader`](flight/core/config_loader.md)
