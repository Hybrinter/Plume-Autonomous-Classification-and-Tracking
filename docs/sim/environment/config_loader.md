# sim.environment.config_loader

**Source:** `packages/sim/src/sim/environment/config_loader.py`
**Kind:** module

## Purpose

The loader reads an environment TOML file into `EnvironmentConfig`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `load_environment_config` | function | Parse a TOML path |

## Inputs and outputs

**`load_environment_config(path) -> Result[EnvironmentConfig, str]`**

- Input: filesystem path.
- Output: `Ok(EnvironmentConfig)` or `Err` with a missing-file, TOML, or
  validation message.

## Behavior

1. The function opens the path with stdlib `tomllib`.
2. A top-level `[environment]` table is unwrapped when present.
3. It validates the mapping with `parse_environment_config`.

## Errors and faults

`Err` for missing file, TOML decode failure, or schema failure.

## Messages

None.

## Configuration

Reads `packages/sim/config/environment.toml` in tests. Callers pass any path.

## Constraints

Flight `config/default.toml` does not contain this table.

## Related documents

- [`sim.environment.config`](config.md)
