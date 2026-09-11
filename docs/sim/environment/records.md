# sim.environment.records

**Source:** `packages/sim/src/sim/environment/records.py`
**Kind:** module

## Purpose

The records module defines frozen dataclasses for one environment evaluation.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `EnvTime` | class | Monotonic step time and UTC seconds |
| `ShutterPose` | class | True gimbal elevation, rate, exposure, gain |
| `PlumeState` | class | ECEF or band-plane plume sample |
| `LookAngles` | class | Mount look angles, SI radians and meters |
| `SceneGeometry` | class | ISS, plume, look, projected centroid |
| `EnvTruth` | class | Oracle record |
| `DriverFeed` | class | Optional mosaic and mask |
| `EnvSample` | class | Truth plus feed |
| `camera_from_sensor` | function | Band-plane `CameraGeometry` from `SensorConfig` |

## Inputs and outputs

**`EnvTime.from_step(clock, now) -> EnvTime`**

- Inputs: injected `Clock`, step monotonic `now`.
- Output: `utc_s = clock.utc_s() + (now - clock.monotonic_s())`.

**`camera_from_sensor(sensor) -> CameraGeometry`**

- Input: `SensorConfig`.
- Output: band-plane size, twice mosaic pitch, focal length in meters.

## Behavior

1. `from_step` maps SIL `now` onto UTC while `ManualClock` still lags by `dt`.
2. `LookAngles.el_rad` is 0 at geocentric nadir and positive along-track.

## Errors and faults

None.

## Messages

None.

## Configuration

None.

## Constraints

- Truth records do not go on the bus.
- `ShutterPose` uses true plant elevation, not encoder counts.

## Related documents

- [`sim.environment`](../environment.md)
- [`sim.environment.evaluate`](evaluate.md)
