# sim.trial

**Source:** `packages/sim/src/sim/trial/`
**Kind:** package

## Purpose

The trial package runs independent open-loop `Environment.evaluate` series. It
does not own world physics. It does not run flight apps.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`spec`](trial/spec.md) | module | `TrialSpec` and rewind shutter params |
| [`seeds`](trial/seeds.md) | module | `spawn_trial_rngs` |
| [`open_loop`](trial/open_loop.md) | module | `run_open_loop`, `StepRecord`, `TrialRecord` |

## Package interface

`sim.trial.__init__` re-exports:

| Name | Kind |
| --- | --- |
| `TrialSpec` | class |
| `RewindThenLimbParams` | class |
| `spawn_trial_rngs` | function |
| `StepRecord` | class |
| `TrialRecord` | class |
| `run_open_loop` | function |
| `shutter_pose_at` | function |

## Interactions

The package imports `sim.environment` and flight `SensorConfig` / `EphemerisConfig`.
It does not import analysis, tools, or gse.

## Constraints

- One `evaluate` call per step. There is no nested Monte Carlo inside evaluate.
- Records hold `EnvTruth` only. They do not hold mosaics or bus telemetry.
- Default CI and GSE do not call `run_open_loop`.

## Related documents

- [`sim`](../sim.md)
- [`sim.environment`](environment.md)
