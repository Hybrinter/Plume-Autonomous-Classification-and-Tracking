# sim.environment

**Source:** `packages/sim/src/sim/environment/`
**Kind:** package

## Purpose

The environment package composes named world models and evaluates them at a
shutter. It produces oracle truth and an optional driver feed. It does not run
flight apps.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`config`](environment/config.md) | module | `EnvironmentConfig` and `build_models` |
| [`config_loader`](environment/config_loader.md) | module | TOML loader for environment files |
| [`evaluate`](environment/evaluate.md) | module | `evaluate_environment` pipeline |
| [`look`](environment/look.md) | module | Mount look angles from ISS to CoG |
| [`records`](environment/records.md) | module | Frozen time, pose, truth, and feed records |
| [`models`](environment/models.md) | package | Named Earth, orbit, wind, plume, optics, appearance models |

## Package interface

`sim.environment.__init__` re-exports:

| Name | Kind |
| --- | --- |
| `Environment` | class |
| `EnvironmentConfig` | class |
| `build_environment` | function |
| `camera_from_sensor` | function |

## Interactions

The package imports flight types, `geo`, `intersect`, and `IssState`. It does
not use the message bus. Flight does not import `sim.environment`.

## Constraints

- Models do not read a clock. Callers pass `EnvTime.from_step`.
- `EnvironmentConfig` is not a field of `PactConfig`.
- Default CI SIL uses `sim.scene.plume` frames unless a harness attaches
  `SilEnvironmentBind`.

## Related documents

- [`sim`](../sim.md)
- [`sim.scene`](scene.md)
- [`sim.sil`](sil.md)
