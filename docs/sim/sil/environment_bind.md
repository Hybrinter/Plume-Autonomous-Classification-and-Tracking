# sim.sil.environment_bind

**Source:** `packages/sim/src/sim/sil/environment_bind.py`
**Kind:** module

## Purpose

The bind evaluates named world models immediately before each SIL cycle and
pushes mosaics and masks into sim drivers.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SilEnvironmentBind` | class | `pre_step(now)` evaluate and driver feed |
| `bind_sil_environment` | function | Construct a bind and reject empty-frame stalls |

## Inputs and outputs

**`bind_sil_environment(environment, sensor, gimbal, clock, sensor_cfg, frames, ...)`**

- Inputs: `Environment`, `SimSensor`, `SimGimbal`, `Clock`, `SensorConfig`,
  constructor frame list, optional `ScriptedDetector`, optional `IssEphemeris`,
  optional RNG.
- Output: `SilEnvironmentBind`.
- Raises `ValueError` when `frames` is empty and a probe evaluate emits no mosaic.

**`SilEnvironmentBind.pre_step(now) -> EnvSample`**

- Input: monotonic seconds for this cycle.
- Output: `EnvSample`. Updates `last_sample` and `last_hal_iss`.

## Behavior

1. `pre_step` calls `gimbal.read_position` so the plant integrates to the clock.
2. It builds `ShutterPose` from `true_el_deg` and `true_omega_rad_s`.
3. `EnvTime.from_step` maps `now` onto UTC.
4. `environment.evaluate` returns truth and a driver feed.
5. A non-None mosaic is wrapped in `MosaicFrame` with `timestamp_s=now` and
   pushed through `SimSensor.load_next`.
6. A non-None mask is pushed through `ScriptedDetector.load_mask` when a
   detector was supplied.
7. HAL `IssEphemeris.read_state` is stored on `last_hal_iss`. It is not a
   `DriverFeed` field.

## Errors and faults

`ValueError` at bind construction when scripted frames are empty and appearance
emits no mosaic.

## Messages

None. The following `step_once` publishes as usual.

## Configuration

Reads `SensorConfig.initial_exposure_us` and `initial_gain_db` for `ShutterPose`.

## Constraints

- Opt-in. Default SIL and GSE leave `bind=None` and keep `sim.scene.plume` frames.
- The bind does not pass `None` into `load_next` or `load_mask`.
- Environment objects still do not own a clock, bus, or gimbal.

## Related documents

- [`sim.sil`](sil.md)
- [`sim.sil.runner`](runner.md)
- [`sim.sil.validation`](validation.md)
- [`sim.environment`](../environment.md)
