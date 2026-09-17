# sim.environment.models.wind

**Source:** `packages/sim/src/sim/environment/models/wind.py`
**Kind:** module

## Purpose

The wind module supplies the `WindModel` Protocol and named still and constant
ECEF implementations.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `WindModel` | Protocol | `velocity_ecef_m_s(utc_s, pos_ecef_m)` |
| `StillWind` | class | Zero wind |
| `ConstantEcefWind` | class | Frozen ECEF velocity |

## Inputs and outputs

**`WindModel.velocity_ecef_m_s(utc_s, pos_ecef_m) -> (vx, vy, vz)`**

- Inputs: UTC seconds and ECEF position metres.
- Output: ECEF velocity metres per second.

## Behavior

1. `StillWind` returns `(0, 0, 0)`.
2. `ConstantEcefWind` returns the stored vector.

## Errors and faults

None.

## Messages

None.

## Configuration

`EnvironmentConfig.constant_ecef` supplies `vx_m_s`, `vy_m_s`, `vz_m_s` when
`wind` is `constant_ecef`.

## Constraints

v1 plume models do not advect. Wind is sampled and unused.

## Related documents

- [`sim.environment.models`](../models.md)
- [`sim.environment.models.plume`](plume.md)
