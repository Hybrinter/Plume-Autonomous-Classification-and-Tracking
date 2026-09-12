# sim.trial.spec

**Source:** `packages/sim/src/sim/trial/spec.py`
**Kind:** module

## Purpose

The spec module defines the open-loop trial recipe.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ShutterKind` | type | `constant` or `rewind_then_limb` |
| `RewindThenLimbParams` | class | Start elevation, limb stop, imaging slew rate |
| `TrialSpec` | class | Trial count, seed, world config, steps, shutter |

## Inputs and outputs

None. This module holds data classes.

## Behavior

1. `TrialSpec.world` is `EnvironmentConfig`.
2. `shutter="constant"` uses `true_el_rad`.
3. `shutter="rewind_then_limb"` uses `RewindThenLimbParams`.

## Errors and faults

Pydantic rejects extra keys, non-positive `dt_s`, and non-positive
`omega_img_rad_s`.

## Messages

None.

## Configuration

None. This module *is* the trial schema.

## Constraints

`TrialSpec` is not part of `PactConfig`.

## Related documents

- [`sim.trial`](../trial.md)
- [`sim.environment.config`](../environment/config.md)
