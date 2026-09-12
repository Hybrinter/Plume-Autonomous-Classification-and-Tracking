"""Frozen records for one environment evaluation at a shutter.

EnvTime maps SIL step monotonic time onto UTC the same way the payload maps
ephemeris reads. ShutterPose is true gimbal optics, not encoder feedback.
EnvTruth is the oracle; DriverFeed is what a SIL bind may push into sim drivers.

Contains:
  - EnvTime, ShutterPose, PlumeState, LookAngles, SceneGeometry
  - EnvTruth, DriverFeed, EnvSample
  - camera_from_sensor
  - PlumeState.present is explicit availability (not inferred from CoG)
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass
from typing import Literal

# third-party
import numpy as np

# internal
from flight.hal.interfaces.ephemeris import IssState
from flight.libs.config import SensorConfig
from flight.libs.time import Clock
from flight.payload.gimbal.intersect import CameraGeometry


@dataclass(frozen=True, slots=True)
class EnvTime:
    """Monotonic shutter time and the matching UTC seconds."""

    monotonic_s: float
    utc_s: float

    @classmethod
    def from_step(cls, clock: Clock, now: float) -> EnvTime:
        """Map SIL step ``now`` onto UTC while the ManualClock still lags by dt.

        Args:
            clock: Injected clock (monotonic and UTC as of the last advance).
            now: Step monotonic seconds passed to step_once.

        Returns:
            EnvTime with monotonic_s=now and utc offset like payload ephemeris reads.
        """
        utc = clock.utc_s() + (now - clock.monotonic_s())
        return cls(monotonic_s=now, utc_s=utc)


@dataclass(frozen=True, slots=True)
class ShutterPose:
    """True gimbal optics at shutter. Not encoder quantization."""

    true_el_rad: float
    true_el_rate_rad_s: float
    exposure_us: float
    gain_db: float


@dataclass(frozen=True, slots=True)
class PlumeState:
    """Plume column or band-plane blob used by look, project, and appearance."""

    frame: Literal["ecef", "bandplane"]
    cog_ecef_m: tuple[float, float, float] | None
    centroid_band_px: tuple[float, float] | None
    along_sigma_m: float
    cross_sigma_m: float
    height_proxy_m: float
    present: bool


@dataclass(frozen=True, slots=True)
class LookAngles:
    """Look from ISS to the plume CoG in the flight mount convention.

    Elevation is 0 rad at geocentric nadir and positive along-track toward the
    limb. Azimuth is optical (unactuated). Slant is meters.
    """

    az_rad: float
    el_rad: float
    eta_rad: float
    slant_m: float
    incidence_rad: float
    visible: bool


@dataclass(frozen=True, slots=True)
class SceneGeometry:
    """ISS, plume, look, and projected centroid between truth and appearance."""

    iss: IssState
    plume: PlumeState
    look: LookAngles
    centroid_band_px: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class EnvTruth:
    """Oracle sample. Never published on the bus."""

    time: EnvTime
    iss: IssState
    plume: PlumeState
    look: LookAngles
    centroid_band_px: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class DriverFeed:
    """Optional mosaic and mask a SIL bind may push into sim drivers."""

    mosaic: np.ndarray | None
    mask: np.ndarray | None


@dataclass(frozen=True, slots=True)
class EnvSample:
    """Truth plus driver feed from one evaluate call."""

    truth: EnvTruth
    feed: DriverFeed


def camera_from_sensor(sensor: SensorConfig) -> CameraGeometry:
    """Build band-plane CameraGeometry from a SensorConfig.

    Args:
        sensor: Flight sensor config (mosaic size, pixel pitch, focal length).

    Returns:
        CameraGeometry at demosaiced band-plane pitch (2x mosaic pitch).
    """
    return CameraGeometry(
        width_px=sensor.width_px // 2,
        height_px=sensor.height_px // 2,
        pixel_pitch_m=2.0 * sensor.pixel_um * 1.0e-6,
        focal_length_m=sensor.focal_length_mm * 1.0e-3,
    )
