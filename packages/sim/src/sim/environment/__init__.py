"""Environment: named world models evaluated at a shutter. No clock, bus, or gimbal."""

from __future__ import annotations

# third-party
import numpy as np

# internal
from flight.libs.config import EphemerisConfig
from flight.libs.types import Err, Ok, Result
from flight.payload.gimbal.intersect import CameraGeometry

from sim.environment.config import EnvironmentConfig, build_models
from sim.environment.evaluate import EnvironmentModels, evaluate_environment
from sim.environment.records import EnvSample, EnvTime, PlumeState, ShutterPose, camera_from_sensor

__all__ = [
    "Environment",
    "EnvironmentConfig",
    "build_environment",
    "camera_from_sensor",
]


class Environment:
    """Composed world models plus camera geometry.

    evaluate does not read a clock. The caller passes EnvTime.from_step.
    """

    def __init__(
        self,
        models: EnvironmentModels,
        camera: CameraGeometry,
        eph: EphemerisConfig,
    ) -> None:
        """Store model bundle, optics, and ephemeris constants."""
        self._models = models
        self._camera = camera
        self._eph = eph

    def evaluate(
        self,
        time: EnvTime,
        shutter: ShutterPose,
        rng: np.random.Generator,
        prior_plume: PlumeState | None = None,
    ) -> EnvSample:
        """Return truth and driver feed at this shutter."""
        return evaluate_environment(
            self._models, self._camera, self._eph, time, shutter, rng, prior_plume
        )


def build_environment(
    config: EnvironmentConfig,
    camera: CameraGeometry,
    eph: EphemerisConfig | None = None,
) -> Result[Environment, str]:
    """Construct an Environment from named models.

    Args:
        config: Named world-model selection.
        camera: Band-plane pinhole geometry (from SensorConfig at bind time).
        eph: Orbit/WGS-84 constants. Defaults to EphemerisConfig().

    Returns:
        Ok(Environment) or Err if a name cannot be resolved.
    """
    cfg = eph if eph is not None else EphemerisConfig()
    models = build_models(config, cfg)
    if isinstance(models, Err):
        return models
    return Ok(Environment(models.value, camera, cfg))
