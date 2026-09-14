"""Thin evaluate pipeline: orbit, plume, look, project, appearance."""

from __future__ import annotations

from dataclasses import dataclass

# third-party
import numpy as np

# internal
from flight.hal.interfaces.ephemeris import IssState
from flight.libs.config import EphemerisConfig
from flight.payload.gimbal.intersect import CameraGeometry

from sim.environment.look import earth_occludes_cog, look_angles_at
from sim.environment.models.appearance import AppearanceModel
from sim.environment.models.earth import EarthModel
from sim.environment.models.optics import OpticsModel
from sim.environment.models.orbit import OrbitModel
from sim.environment.models.plume import PlumeModel
from sim.environment.models.wind import WindModel
from sim.environment.records import (
    EnvSample,
    EnvTime,
    EnvTruth,
    LookAngles,
    PlumeState,
    SceneGeometry,
    ShutterPose,
)


@dataclass(frozen=True, slots=True)
class EnvironmentModels:
    """Protocol bundle selected by EnvironmentConfig."""

    earth: EarthModel
    orbit: OrbitModel
    wind: WindModel
    plume: PlumeModel
    optics: OpticsModel
    appearance: AppearanceModel


def evaluate_environment(
    models: EnvironmentModels,
    camera: CameraGeometry,
    eph: EphemerisConfig,
    time: EnvTime,
    shutter: ShutterPose,
    rng: np.random.Generator,
    prior_plume: PlumeState | None,
) -> EnvSample:
    """Compose one truth sample and driver feed. Infallible after build_environment.

    Args:
        models: Named-model instances.
        camera: Band-plane pinhole geometry.
        eph: WGS-84 / epoch constants (same as HAL ephemeris config).
        time: Step time (from_step).
        shutter: True gimbal pose.
        rng: Per-trial generator (plume draws and appearance noise).
        prior_plume: Previous PlumeState for PoissonLatitude and later advection.

    Returns:
        EnvSample with EnvTruth and DriverFeed.
    """
    iss = models.orbit.state_eci(time.utc_s)
    plume = models.plume.evaluate(
        time,
        prior_plume,
        models.wind,
        models.earth,
        iss,
        models.orbit,
        eph.omega_earth_rad_s,
        eph.epoch_utc_s,
        rng,
    )
    geom = _build_scene_geometry(models, camera, eph, iss, plume, shutter)
    truth = EnvTruth(
        time=time,
        iss=geom.iss,
        plume=geom.plume,
        look=geom.look,
        centroid_band_px=geom.centroid_band_px,
    )
    feed = models.appearance.render_feed(geom, shutter, camera, rng)
    return EnvSample(truth=truth, feed=feed)


def _build_scene_geometry(
    models: EnvironmentModels,
    camera: CameraGeometry,
    eph: EphemerisConfig,
    iss: IssState,
    plume: PlumeState,
    shutter: ShutterPose,
) -> SceneGeometry:
    """Look and project, or copy band-plane centroid without ECEF."""
    if plume.frame == "bandplane":
        look = LookAngles(0.0, 0.0, 0.0, 0.0, 0.0, False)
        return SceneGeometry(
            iss=iss,
            plume=plume,
            look=look,
            centroid_band_px=plume.centroid_band_px,
        )
    look = LookAngles(0.0, 0.0, 0.0, 0.0, 0.0, False)
    centroid: tuple[float, float] | None = None
    if plume.cog_ecef_m is not None:
        look = look_angles_at(
            iss,
            plume.cog_ecef_m,
            models.earth,
            eph.omega_earth_rad_s,
            eph.epoch_utc_s,
        )
        if not earth_occludes_cog(
            iss,
            plume.cog_ecef_m,
            models.earth,
            eph.omega_earth_rad_s,
            eph.epoch_utc_s,
        ):
            centroid = models.optics.project_centroid(
                iss,
                shutter,
                plume.cog_ecef_m,
                camera,
                eph.omega_earth_rad_s,
                eph.epoch_utc_s,
            )
    return SceneGeometry(iss=iss, plume=plume, look=look, centroid_band_px=centroid)
