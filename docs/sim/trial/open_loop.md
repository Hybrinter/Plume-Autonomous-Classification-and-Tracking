# sim.trial.open_loop

**Source:** `packages/sim/src/sim/trial/open_loop.py`
**Kind:** module

## Purpose

The open-loop module steps `Environment.evaluate` and records oracle truth.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `StepRecord` | class | Step index, shutter, `EnvTruth` |
| `TrialRecord` | class | Trial id and step tuple |
| `shutter_pose_at` | function | Scripted 1-axis `ShutterPose` |
| `run_open_loop` | function | Independent trials |

## Inputs and outputs

**`run_open_loop(spec, sensor=None, eph=None) -> tuple[TrialRecord, ...]`**

- Inputs: `TrialSpec`, optional `SensorConfig`, optional `EphemerisConfig`.
- Output: one `TrialRecord` per trial.

**`shutter_pose_at(spec, now) -> ShutterPose`**

- Inputs: `TrialSpec`, monotonic seconds.
- Output: elevation, rate, exposure, gain.

## Behavior

1. `spawn_trial_rngs` supplies one Generator per trial.
2. Each trial builds an `Environment` from `spec.world`.
3. Each step advances `now` by `dt_s`, evaluates, and threads `prior_plume`.
4. Rewind shutter slews from `el_start_rad` to `el_limb_rad` and then holds.

## Errors and faults

`ValueError` when `steps` is not positive or `build_environment` returns `Err`.

## Messages

None.

## Configuration

None.

## Constraints

- No bus, SIL bind, or `CaptureResult`.
- Elevation only. There is no azimuth plant.

## Related documents

- [`sim.trial`](../trial.md)
- [`sim.environment.evaluate`](../environment/evaluate.md)
- [`sim.environment.records`](../environment/records.md)
