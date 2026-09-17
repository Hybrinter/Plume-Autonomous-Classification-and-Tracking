"""Pinhole projection: ECEF CoG to band-plane pixels via flight camera/mount frames."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# third-party
import numpy as np

# internal
from flight.hal.interfaces.ephemeris import IssState
from flight.payload.gimbal.geo import eci_from_ecef, lvlh_axes, ry
from flight.payload.gimbal.intersect import CameraGeometry

from sim.environment.records import ShutterPose


@runtime_checkable
class OpticsModel(Protocol):
    """Project an ECEF point into the band plane at the shutter pose."""

    def project_centroid(
        self,
        iss: IssState,
        shutter: ShutterPose,
        cog_ecef_m: tuple[float, float, float],
        camera: CameraGeometry,
        omega_earth_rad_s: float,
        epoch_utc_s: float,
    ) -> tuple[float, float] | None:
        """Return band-plane (u, v) or None if behind the camera."""
        ...


@dataclass(frozen=True, slots=True)
class PinholeOptics:
    """Inverse of flight pinhole_cam_ray + cam_ray_to_mount + mount_to_eci."""

    def project_centroid(
        self,
        iss: IssState,
        shutter: ShutterPose,
        cog_ecef_m: tuple[float, float, float],
        camera: CameraGeometry,
        omega_earth_rad_s: float,
        epoch_utc_s: float,
    ) -> tuple[float, float] | None:
        """Project ECEF CoG into band-plane pixels at true elevation."""
        r_iss = np.asarray(iss.r_m, dtype=np.float64)
        v_iss = np.asarray(iss.v_m_s, dtype=np.float64)
        cog_ecef = np.asarray(cog_ecef_m, dtype=np.float64)
        cog_eci = eci_from_ecef(cog_ecef, omega_earth_rad_s, iss.epoch_utc_s, epoch_utc_s)
        look_eci = cog_eci - r_iss
        slant = float(np.linalg.norm(look_eci))
        if slant < 1.0:
            return None
        d_eci = look_eci / slant
        x_hat, y_hat, z_hat = lvlh_axes(r_iss, v_iss)
        d_mount = np.array(
            [float(np.dot(d_eci, x_hat)), float(np.dot(d_eci, y_hat)), float(np.dot(d_eci, z_hat))],
            dtype=np.float64,
        )
        d_nadir = ry(-shutter.true_el_rad) @ d_mount
        # Inverse of cam_ray_to_mount at nadir: d_nadir = (-d_cam_y, d_cam_x, d_cam_z)
        d_cam = np.array(
            [float(d_nadir[1]), -float(d_nadir[0]), float(d_nadir[2])],
            dtype=np.float64,
        )
        if d_cam[2] <= 1e-12:
            return None
        u0 = camera.width_px / 2.0
        v0 = camera.height_px / 2.0
        # Inverse of pinhole_cam_ray: ray = ((u-u0)*pitch/f, (v-v0)*pitch/f, 1)
        u_px = u0 + camera.focal_length_m * d_cam[0] / (d_cam[2] * camera.pixel_pitch_m)
        v_px = v0 + camera.focal_length_m * d_cam[1] / (d_cam[2] * camera.pixel_pitch_m)
        if not math.isfinite(u_px) or not math.isfinite(v_px):
            return None
        return (float(u_px), float(v_px))
