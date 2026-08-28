# flight.core.config_loader

**Source:** `packages/flight/src/flight/core/config_loader.py`
**Kind:** module

## Purpose

The config loader merges TOML files and validates them into the frozen `PactConfig` schema.
No subsystem reads TOML directly.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `load_config` | function | Load, merge, validate, and map TOML into `PactConfig` |

## Inputs and outputs

**`load_config(config_path, override_path=None) -> Result[PactConfig, str]`**

- Inputs: path to the base TOML file; optional override path.
- Output: `Ok(PactConfig)` on success, or `Err(str)` with a human-readable error.

## Behavior

1. Load the base TOML file at `config_path`.
2. When `override_path` is set, load the override and deep-merge it on top of the base.
3. Validate the merged dict with the `PactConfig` schema.
4. Return `Ok(config)` or `Err` on any failure.

The schema rejects unknown sections and unknown keys. Field constraints cover numeric
ranges, APID width, port bounds, even mosaic dimensions, and non-empty strings. Cross-field
rules cover the gimbal travel envelope, stow and home poses, mosaic layout permutation,
and inference input-band membership.

## Errors and faults

Returns `Err(str)` for:

- Missing or unreadable config or override file
- TOML parse error
- Unknown section or key
- Out-of-range field value
- Cross-field violation (gimbal bounds, mosaic layout, input bands)
- Schema validation error

## Messages

None.

## Configuration

Reads all TOML sections backed by the `PactConfig` schema:

| Section | Dataclass |
| --- | --- |
| `controller` | `ControllerConfig` |
| `inference` | `InferenceConfig` |
| `comms` | `CommsConfig` |
| `storage` | `StorageConfig` |
| `fault` | `FaultConfig` |
| `preprocessing` | `PreprocessingConfig` |
| `sensor` | `SensorConfig` |
| `gimbal` | `GimbalConfig` |
| `link` | `LinkConfig` |
| `command_ingress` | `CommandIngressConfig` |
| `command_router` | `CommandRouterConfig` |
| `environment` | `EnvironmentConfig` |

## Constraints

- Override values replace base values at every nesting level.
- Unknown keys fail at startup. They are not silently ignored.
- APID fields must fit in 11 bits. Port fields must be in 1..65535.
- Sensor width and height must be even positive integers.
- When `command_ingress.require_auth` is true, `hmac_key_path` must be non-empty.

## Related documents

- [`flight.core`](../core.md)
- [`flight.core.main`](main.md)
- [`flight.libs.config`](../libs/config.md)
