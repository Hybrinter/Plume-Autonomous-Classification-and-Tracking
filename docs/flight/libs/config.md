# flight.libs.config

**Source:** `packages/flight/src/flight/libs/config/`
**Kind:** package

## Purpose

The config package holds frozen dataclasses for every tunable flight parameter. The config
loader validates TOML into these types. Subsystems receive typed config slices; they do not
read TOML directly.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`config`](config/config.md) | module | Per-subsystem config dataclasses and `PactConfig` |

## Package interface

`flight.libs.config` re-exports:

| Name | Kind |
| --- | --- |
| `AxisMode` | type alias |
| `CommandIngressConfig`, `CommandRouterConfig`, `CommsConfig` | class |
| `ControllerConfig`, `EnvironmentConfig`, `FaultConfig` | class |
| `GimbalConfig`, `InferenceConfig`, `LinkConfig` | class |
| `PactConfig`, `PreprocessingConfig`, `SensorConfig`, `StorageConfig` | class |
| `ThermalConfig` | class |

## Interactions

None at the package level. The composition root loads `PactConfig` and passes each subsystem
its sub-config at construction time.

## Constraints

- All config dataclasses are frozen.
- Default field values must match `config/default.toml` exactly.
- `packages/flight/tests/libs/config/test_config_defaults.py` asserts TOML and Python
  defaults stay equal.
- Tuple defaults for array-like fields are compared after list-to-tuple normalization in tests.

## Related documents

- [`flight.libs`](../libs.md)
- [`flight.libs.config.config`](config/config.md)
