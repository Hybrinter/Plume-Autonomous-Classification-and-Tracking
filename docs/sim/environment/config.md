# sim.environment.config

**Source:** `packages/sim/src/sim/environment/config.py`
**Kind:** module

## Purpose

The config module names world models and builds the Protocol bundle.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `EnvironmentConfig` | class | Axis names plus optional param blocks |
| `parse_environment_config` | function | Validate a dict |
| `build_models` | function | Name string to Protocol instances |

## Inputs and outputs

**`parse_environment_config(data) -> Result[EnvironmentConfig, str]`**

- Input: mapping of axis names and nested param tables.
- Output: `Ok` config or `Err` validation text.

**`build_models(config, eph) -> Result[EnvironmentModels, str]`**

- Inputs: `EnvironmentConfig`, `EphemerisConfig`.
- Output: `Ok(EnvironmentModels)`.

## Behavior

1. Axis fields are `Literal` unions (`wgs84_ellipsoid`, `circular_kepler`, ...).
2. `build_models` constructs Earth, orbit, wind, plume, optics, and appearance.
3. `poisson_latitude` table lengths must match and latitudes must increase.

## Errors and faults

`Err` on pydantic validation failure. Unknown names cannot pass the Literal
types. Unequal or unsorted `poisson_latitude` tables return `Err`.

## Messages

None.

## Configuration

None. This module *is* the environment config schema.

## Constraints

- `EnvironmentConfig` is not part of `PactConfig`.
- Extra TOML keys are forbidden.

## Related documents

- [`sim.environment`](../environment.md)
- [`sim.environment.config_loader`](config_loader.md)
