"""Open-loop evaluate series. No bus, no SIL bind, no CaptureResult."""

from __future__ import annotations

from dataclasses import dataclass

from flight.libs.config import EphemerisConfig, SensorConfig
from flight.libs.types import Err

from sim.environment import Environment, build_environment, camera_from_sensor
from sim.environment.records import EnvTime, EnvTruth, ShutterPose
from sim.trial.seeds import spawn_trial_rngs
from sim.trial.spec import TrialSpec


@dataclass(frozen=True, slots=True)
class StepRecord:
    """One evaluate sample: shutter pose and oracle truth."""

    step_index: int
    shutter: ShutterPose
    truth: EnvTruth


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """One independent open-loop trial."""

    trial_id: int
    steps: tuple[StepRecord, ...]


def shutter_pose_at(spec: TrialSpec, now: float) -> ShutterPose:
    """Return the scripted 1-axis shutter at monotonic now.

    Args:
        spec: Trial recipe (constant or rewind-then-limb).
        now: Seconds from trial start.

    Returns:
        ShutterPose. Azimuth is not actuated.
    """
    if spec.shutter == "constant":
        return ShutterPose(spec.true_el_rad, 0.0, spec.exposure_us, spec.gain_db)
    rw = spec.rewind
    delta = rw.el_limb_rad - rw.el_start_rad
    t_slew = abs(delta) / rw.omega_img_rad_s
    sign = 1.0 if delta >= 0.0 else -1.0
    if now < t_slew:
        el = rw.el_start_rad + sign * rw.omega_img_rad_s * now
        return ShutterPose(el, sign * rw.omega_img_rad_s, spec.exposure_us, spec.gain_db)
    return ShutterPose(rw.el_limb_rad, 0.0, spec.exposure_us, spec.gain_db)


def run_open_loop(
    spec: TrialSpec,
    sensor: SensorConfig | None = None,
    eph: EphemerisConfig | None = None,
) -> tuple[TrialRecord, ...]:
    """Evaluate the world for n_trials independent RNG streams.

    Args:
        spec: Trial recipe. steps must be positive.
        sensor: Camera geometry. Defaults to SensorConfig().
        eph: Orbit/WGS-84 constants for UTC mapping. Defaults to EphemerisConfig().

    Returns:
        One TrialRecord per trial, each with spec.steps StepRecords.

    Raises:
        ValueError: steps is not positive, or build_environment fails.
    """
    if spec.steps <= 0:
        raise ValueError(f"steps must be positive, got {spec.steps}")
    used_sensor = sensor if sensor is not None else SensorConfig()
    used_eph = eph if eph is not None else EphemerisConfig()
    camera = camera_from_sensor(used_sensor)
    rngs = spawn_trial_rngs(spec.master_seed, spec.n_trials)
    records: list[TrialRecord] = []
    for trial_id, rng in enumerate(rngs):
        built = build_environment(spec.world, camera, used_eph)
        if isinstance(built, Err):
            raise ValueError(built.error)
        env: Environment = built.value
        prior = None
        step_rows: list[StepRecord] = []
        now = 0.0
        for step_index in range(1, spec.steps + 1):
            now += spec.dt_s
            time = EnvTime(monotonic_s=now, utc_s=used_eph.epoch_utc_s + now)
            shutter = shutter_pose_at(spec, now)
            sample = env.evaluate(time, shutter, rng, prior)
            prior = sample.truth.plume
            step_rows.append(StepRecord(step_index, shutter, sample.truth))
        records.append(TrialRecord(trial_id, tuple(step_rows)))
    return tuple(records)
