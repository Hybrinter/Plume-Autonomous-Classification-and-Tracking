# flight.core.config_loader

**Source:** `packages/flight/src/flight/core/config_loader.py`
**Kind:** module

## Purpose

The config loader merges TOML files and maps them into the frozen `PactConfig` hierarchy.
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
3. Run `_validate` on the merged dict.
4. Map the merged dict to `PactConfig` via `_build_pact_config`.
5. Return `Ok(config)` or `Err` on any failure.

**Validation steps in `_validate`:**

1. Reject unknown top-level sections and unknown keys within each section.
2. Check per-section numeric ranges for controller, inference, sensor, fault, storage,
   comms, preprocessing, link, command_ingress, and command_router fields.
3. Check cross-field rules: gimbal travel envelope, stow/home pose bounds, mosaic layout
   permutation, and inference input band membership.

**Mapping:** Each TOML section maps to one frozen dataclass in `flight.libs.config`.
Environment axes resolve to `"sim"` or `"real"` literals.

## Errors and faults

Returns `Err(str)` for:

- Missing or unreadable config or override file
- TOML parse error
- Unknown section or key
- Out-of-range field value
- Cross-field violation (gimbal bounds, mosaic layout, input bands)
- Mapping error (`KeyError`, `TypeError`, `ValueError`)

## Messages

None.

## Configuration

Reads all TOML sections backed by `_SECTION_TO_CLASS`:

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
