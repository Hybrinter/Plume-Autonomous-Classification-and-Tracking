"""Opt-in SIL bind: evaluate the world, then feed sim drivers, then step_once.

SilEnvironmentBind does not own a clock, bus, or gimbal plant. pre_step
advances the plant without an encoder sample, evaluates the environment, and
pushes only non-None mosaics and masks. HAL ephemeris is logged beside truth
for scoring and never placed in DriverFeed. Live mosaics are refused while
constructor frames remain so frame_id values cannot collide.

Contains:
  - SilEnvironmentBind: per-step evaluate + driver feed
  - bind_sil_environment: construct and reject empty-frame + no-mosaic,
    and reject constructor frames mixed with appearance mosaics
"""

from __future__ import annotations

# stdlib
import math

# third-party
import numpy as np

# internal
from flight.hal.drivers_sim import SimGimbal, SimSensor
from flight.hal.interfaces.ephemeris import IssEphemeris, IssState
from flight.libs.config import SensorConfig
from flight.libs.time import Clock
from flight.libs.types import MosaicFrame, Ok
from flight.payload.inference import ScriptedDetector

from sim.environment import Environment
from sim.environment.records import EnvSample, EnvTime, PlumeState, ShutterPose


class SilEnvironmentBind:
    """Evaluate the simulated world immediately before each SIL cycle."""

    def __init__(
        self,
        environment: Environment,
        sensor: SimSensor,
        gimbal: SimGimbal,
        clock: Clock,
        sensor_cfg: SensorConfig,
        detector: ScriptedDetector | None = None,
        ephemeris: IssEphemeris | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Store drivers and models. Use bind_sil_environment for the empty-frame check."""
        self._environment = environment
        self._sensor = sensor
        self._gimbal = gimbal
        self._clock = clock
        self._sensor_cfg = sensor_cfg
        self._detector = detector
        self._ephemeris = ephemeris
        self._rng = rng if rng is not None else np.random.default_rng(0)
        self._prior_plume: PlumeState | None = None
        self._frame_id = 0
        self.last_sample: EnvSample | None = None
        self.last_hal_iss: IssState | None = None

    def pre_step(self, now: float) -> EnvSample:
        """Integrate the gimbal plant, evaluate, and push non-None feed slots.

        Args:
            now: Step monotonic seconds (same value later passed to step_once).

        Returns:
            The EnvSample from evaluate. last_sample and last_hal_iss are updated.

        Raises:
            ValueError: Appearance emitted a mosaic while constructor frames remain.
        """
        self._gimbal.advance_plant()
        shutter = ShutterPose(
            true_el_rad=math.radians(self._gimbal.true_el_deg),
            true_el_rate_rad_s=self._gimbal.true_omega_rad_s,
            exposure_us=self._sensor_cfg.initial_exposure_us,
            gain_db=self._sensor_cfg.initial_gain_db,
        )
        time = EnvTime.from_step(self._clock, now)
        sample = self._environment.evaluate(time, shutter, self._rng, self._prior_plume)
        self._prior_plume = sample.truth.plume
        self.last_sample = sample
        self.last_hal_iss = self._read_hal_iss(time.utc_s)
        mosaic = sample.feed.mosaic
        if mosaic is not None:
            if self._sensor.unread_scripted_count() > 0:
                raise ValueError(
                    "appearance mosaics cannot interleave with unread constructor frames"
                )
            self._frame_id += 1
            self._sensor.load_next(
                MosaicFrame(
                    timestamp_utc=self._clock.wall_clock_iso(),
                    timestamp_s=now,
                    frame_id=self._frame_id,
                    mosaic=mosaic,
                    exposure_us=shutter.exposure_us,
                    gain_db=shutter.gain_db,
                )
            )
        mask = sample.feed.mask
        if mask is not None and self._detector is not None:
            self._detector.load_mask(np.asarray(mask, dtype=np.float32))
        return sample

    def _read_hal_iss(self, utc_s: float) -> IssState | None:
        """Read SimIssEphemeris (or any IssEphemeris) beside truth. Not a DriverFeed field."""
        if self._ephemeris is None:
            return None
        result = self._ephemeris.read_state(utc_s)
        if isinstance(result, Ok):
            return result.value
        return None


def bind_sil_environment(
    environment: Environment,
    sensor: SimSensor,
    gimbal: SimGimbal,
    clock: Clock,
    sensor_cfg: SensorConfig,
    frames: list[MosaicFrame],
    detector: ScriptedDetector | None = None,
    ephemeris: IssEphemeris | None = None,
    rng: np.random.Generator | None = None,
) -> SilEnvironmentBind:
    """Construct a bind. Empty scripted frames require an appearance mosaic.

    Args:
        environment: Named-model world.
        sensor: Sim camera that receives load_next.
        gimbal: Sim plant that supplies true elevation.
        clock: Shared SIL clock.
        sensor_cfg: Exposure, gain, and mosaic metadata.
        frames: Constructor frames already given to SimSensor (may be empty).
        detector: Scripted detector for load_mask. None skips mask push.
        ephemeris: HAL ISS source logged beside truth.
        rng: Appearance RNG. Defaults to a seeded Generator.

    Returns:
        A ready SilEnvironmentBind.

    Raises:
        ValueError: frames is empty and a probe evaluate emits no mosaic, or
            frames remain and a probe evaluate emits a mosaic.
    """
    bind = SilEnvironmentBind(
        environment,
        sensor,
        gimbal,
        clock,
        sensor_cfg,
        detector=detector,
        ephemeris=ephemeris,
        rng=rng,
    )
    probe_rng = np.random.default_rng(1)
    probe_time = EnvTime.from_step(clock, clock.monotonic_s())
    probe_shutter = ShutterPose(
        true_el_rad=0.0,
        true_el_rate_rad_s=0.0,
        exposure_us=sensor_cfg.initial_exposure_us,
        gain_db=sensor_cfg.initial_gain_db,
    )
    probe = environment.evaluate(probe_time, probe_shutter, probe_rng, None)
    if len(frames) == 0:
        if probe.feed.mosaic is None:
            raise ValueError(
                "empty SimSensor frames require an appearance model that emits mosaics"
            )
    elif probe.feed.mosaic is not None:
        raise ValueError("constructor frames and appearance mosaics cannot be mixed")
    return bind
