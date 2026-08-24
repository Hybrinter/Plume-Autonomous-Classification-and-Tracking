# flight.libs.config

**Source:** `packages/flight/src/flight/libs/config`
**Kind:** package

## Purpose

This package defines frozen configuration dataclasses for every flight subsystem. Defaults
match `config/default.toml`. Subsystems receive typed config objects; they do not read TOML.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`config`](flight/libs/config/config.md) | module | Per-subsystem config dataclasses and `PactConfig` |

## Package interface

Re-exports from `flight.libs.config.config`:

- `AxisMode`
- `CommandIngressConfig`
- `CommandRouterConfig`
- `CommsConfig`
- `ControllerConfig`
- `EnvironmentConfig`
- `FaultConfig`
- `GimbalConfig`
- `InferenceConfig`
- `LinkConfig`
- `PactConfig`
- `PreprocessingConfig`
- `SensorConfig`
- `StorageConfig`

## Interactions

`flight.core.config_loader.load_config` constructs `PactConfig` from TOML and passes each
subsystem its sub-config at composition time.

## Constraints

- All config dataclasses are frozen.
- Field defaults must match `config/default.toml` exactly.
- No subsystem reads TOML directly.

## Related documents

- [`flight.libs`](flight/libs.md)
- [`flight.libs.config.config`](flight/libs/config/config.md)
- [`flight.core.config_loader`](flight/core/config_loader.md)
