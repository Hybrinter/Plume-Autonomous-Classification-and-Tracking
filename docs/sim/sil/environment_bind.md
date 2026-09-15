# sim.sil.environment_bind

**Source:** `packages/sim/src/sim/sil/environment_bind.py`
**Kind:** module

## Purpose

The bind evaluates named world models at shutter after SIL loop catch-up and
pushes mosaics and masks into sim drivers.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SilEnvironmentBind` | class | `pre_step(now)` evaluate and driver feed |
| `bind_sil_environment` | function | Construct a bind and reject empty-frame or mixed-mosaic cases |

## Inputs and outputs

**`bind_sil_environment(environment, sensor, gimbal, clock, sensor_cfg, frames, ...)`**

- Inputs: `Environment`, `SimSensor`, `SimGimbal`, `Clock`, `SensorConfig`,
  constructor frame list, optional `ScriptedDetector`, optional `IssEphemeris`,
  optional RNG.
- Output: `SilEnvironmentBind`.
- Raises `ValueError` when `frames` is empty and a probe evaluate emits no mosaic.
- Raises `ValueError` when `frames` remain and a probe evaluate emits a mosaic.

**`SilEnvironmentBind.pre_step(now) -> EnvSample`**

- Input: monotonic seconds for this cycle.
- Output: `EnvSample`. Updates `last_sample` and `last_hal_iss`.
- Raises `ValueError` when appearance emits a mosaic and constructor frames remain.

## Behavior

1. `step_once` calls `pre_step` after inner/outer catch-up and before acquire.
2. `pre_step` calls `gimbal.advance_plant`. After catch-up the clock is still
   frozen, so this call leaves catch-up debt in place. `ShutterPose` reads
   `true_el_deg` and `true_omega_rad_s` at that plant state.
3. `EnvTime.from_step` maps `now` onto UTC.
4. `environment.evaluate` returns truth and a driver feed.
5. A non-None mosaic is wrapped in `MosaicFrame` with `timestamp_s=now` and
   pushed through `SimSensor.load_next` when no constructor frames remain.
6. A non-None mask is pushed through `ScriptedDetector.load_mask` when a
   detector was supplied and no constructor frames remain.
7. HAL `IssEphemeris.read_state` is stored on `last_hal_iss`. It is not a
   `DriverFeed` field.

Scene, camera pose, frame metadata, and encoder samples share shutter time `now`.

## Errors and faults

`ValueError` at bind construction when scripted frames are empty and appearance
emits no mosaic. `ValueError` at construction or `pre_step` when appearance
mosaics would mix with unread constructor frames. A live mask with unread
constructor frames is skipped and does not raise.

## Messages

None. The following `step_once` acquire publishes as usual.

## Configuration

Reads `SensorConfig.initial_exposure_us` and `initial_gain_db` for `ShutterPose`.

## Constraints

- Opt-in. Default SIL and GSE leave `bind=None` and keep `sim.scene.plume` frames.
- The bind does not pass `None` into `load_next` or `load_mask`.
- Live mosaics require empty constructor frames. An OracleMask bind may keep
  scripted frames. Those unread frames block `load_mask`.
- Environment objects still do not own a clock, bus, or gimbal.
- `advance_plant` does not sample the encoder.

## Related documents

- [`sim.sil`](sil.md)
- [`sim.sil.runner`](runner.md)
- [`sim.sil.validation`](validation.md)
- [`sim.sil.stepping`](stepping.md)
- [`sim.environment`](../environment.md)
