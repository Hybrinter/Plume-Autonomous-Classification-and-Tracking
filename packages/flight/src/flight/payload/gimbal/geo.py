"""Mount, camera, LVLH, and WGS-84 geometry helpers for elevation pointing (pure).

All functions are side-effect free. Angles are radians unless a name says otherwise.
Vectors are ECI or ECEF meters. Identity mount versus ISS: the mount frame is the
ISS nadir / along-track / starboard triad from the current ECI state.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

# stdlib
import math
from typing import cast

# third-party
import numpy as np

_MIN_RANGE_M = 1.0


def rx_neg(theta_rad: float) -> np.ndarray:
    """Rotation matrix R_x(-theta). Maps mount-nadir boresight to signed elevation.

    Inputs:
        theta_rad: Signed off-nadir elevation (positive along-track).

    Outputs:
        np.ndarray[float64, (3, 3)]: Right-handed rotation about mount +x.
    """
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_t, sin_t], [0.0, -sin_t, cos_t]],
        dtype=np.float64,
    )


def boresight_mount(theta_rad: float) -> np.ndarray:
    """Unit boresight in mount coordinates: (0, sin theta, cos theta).

    Inputs:
        theta_rad: Signed off-nadir elevation.

    Outputs:
        np.ndarray[float64, (3,)]: Mount-frame unit look vector.
    """
    return np.array([0.0, math.sin(theta_rad), math.cos(theta_rad)], dtype=np.float64)


def rz(phi_rad: float) -> np.ndarray:
    """Right-handed rotation about +z by phi.

    Inputs:
        phi_rad: Rotation angle in radians.

    Outputs:
        np.ndarray[float64, (3, 3)]: R_z(phi).
    """
    cos_p = math.cos(phi_rad)
    sin_p = math.sin(phi_rad)
    return np.array(
        [[cos_p, -sin_p, 0.0], [sin_p, cos_p, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def eci_from_ecef(
    r_ecef_m: np.ndarray,
    omega_earth_rad_s: float,
    utc_s: float,
    epoch_utc_s: float,
) -> np.ndarray:
    """Rotate an ECEF vector into ECI. Frames align at epoch_utc_s.

    Inputs:
        r_ecef_m: ECEF meters, shape (3,).
        omega_earth_rad_s: Earth rotation rate.
        utc_s: Current UTC seconds.
        epoch_utc_s: UTC seconds at which ECEF and ECI axes coincide.

    Outputs:
        np.ndarray[float64, (3,)]: ECI meters.
    """
    return cast(np.ndarray, rz(omega_earth_rad_s * (utc_s - epoch_utc_s)) @ r_ecef_m)


def ecef_from_eci(
    r_eci_m: np.ndarray,
    omega_earth_rad_s: float,
    utc_s: float,
    epoch_utc_s: float,
) -> np.ndarray:
    """Rotate an ECI vector into ECEF. Frames align at epoch_utc_s.

    Inputs:
        r_eci_m: ECI meters, shape (3,).
        omega_earth_rad_s: Earth rotation rate.
        utc_s: Current UTC seconds.
        epoch_utc_s: UTC seconds at which ECEF and ECI axes coincide.

    Outputs:
        np.ndarray[float64, (3,)]: ECEF meters.
    """
    return cast(np.ndarray, rz(-omega_earth_rad_s * (utc_s - epoch_utc_s)) @ r_eci_m)


def lvlh_axes(
    r_eci_m: np.ndarray, v_eci_m_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ISS body axes in ECI: +x starboard (orbit normal), +y along-track, +z nadir.

    Inputs:
        r_eci_m: ISS position ECI meters, shape (3,).
        v_eci_m_s: ISS inertial velocity ECI m/s, shape (3,).

    Outputs:
        tuple of three np.ndarray[float64, (3,)]: (x_hat, y_hat, z_hat).
    """
    r_norm = float(np.linalg.norm(r_eci_m))
    h = np.cross(r_eci_m, v_eci_m_s)
    h_norm = float(np.linalg.norm(h))
    if r_norm < _MIN_RANGE_M or h_norm < 1e-12:
        return (
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )
    x_hat = h / h_norm
    z_hat = -r_eci_m / r_norm
    y_hat = np.cross(z_hat, x_hat)
    y_norm = float(np.linalg.norm(y_hat))
    if y_norm < 1e-12:
        y_hat = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        y_hat = y_hat / y_norm
    return x_hat, y_hat, z_hat


def pinhole_cam_ray(
    u_px: float,
    v_px: float,
    plane_width_px: int,
    plane_height_px: int,
    pixel_pitch_m: float,
    focal_m: float,
) -> np.ndarray:
    """Unit look vector in the camera frame from a band-plane pixel.

    Camera +Z is boresight, +X is image right, +Y is image down.

    Inputs:
        u_px, v_px: Band-plane pixel coordinates.
        plane_width_px, plane_height_px: Band-plane size (principal point at center).
        pixel_pitch_m: Band-plane pitch in meters.
        focal_m: Focal length in meters.

    Outputs:
        np.ndarray[float64, (3,)]: Unit camera-frame ray.
    """
    u0 = plane_width_px / 2.0
    v0 = plane_height_px / 2.0
    ray = np.array(
        [(u_px - u0) * pixel_pitch_m / focal_m, (v_px - v0) * pixel_pitch_m / focal_m, 1.0],
        dtype=np.float64,
    )
    nrm = float(np.linalg.norm(ray))
    if nrm < 1e-18:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return ray / nrm


def cam_ray_to_mount(d_cam: np.ndarray, theta_g_rad: float) -> np.ndarray:
    """Rotate a camera-frame ray into the mount frame at elevation theta_g.

    At nadir, camera +X is mount +x and camera +Y is mount -y.

    Inputs:
        d_cam: Unit camera-frame ray, shape (3,).
        theta_g_rad: Gimbal elevation.

    Outputs:
        np.ndarray[float64, (3,)]: Unit mount-frame look vector.
    """
    d_nadir = np.array([float(d_cam[0]), -float(d_cam[1]), float(d_cam[2])], dtype=np.float64)
    nrm = float(np.linalg.norm(d_nadir))
    if nrm < 1e-18:
        d_nadir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        d_nadir = d_nadir / nrm
    return rx_neg(theta_g_rad) @ d_nadir


def mount_to_eci(d_mount: np.ndarray, r_eci_m: np.ndarray, v_eci_m_s: np.ndarray) -> np.ndarray:
    """Express a mount-frame look vector in ECI using the ISS LVLH triad.

    Inputs:
        d_mount: Unit mount-frame vector, shape (3,).
        r_eci_m: ISS ECI position meters.
        v_eci_m_s: ISS ECI velocity m/s.

    Outputs:
        np.ndarray[float64, (3,)]: Unit ECI look vector.
    """
    x_hat, y_hat, z_hat = lvlh_axes(r_eci_m, v_eci_m_s)
    d_eci = d_mount[0] * x_hat + d_mount[1] * y_hat + d_mount[2] * z_hat
    nrm = float(np.linalg.norm(d_eci))
    if nrm < 1e-18:
        return z_hat
    return cast(np.ndarray, d_eci / nrm)


def wgs84_intersect(
    r0_ecef_m: np.ndarray,
    d_ecef: np.ndarray,
    a_m: float,
    f: float,
) -> tuple[np.ndarray, float] | None:
    """Forward intersect of a ray with the WGS-84 ellipsoid.

    Inputs:
        r0_ecef_m: Ray origin ECEF meters (ISS position).
        d_ecef: Unit look direction in ECEF.
        a_m: WGS-84 semi-major axis meters.
        f: WGS-84 flattening.

    Outputs:
        (r_hit_ecef_m, slant_m) for the nearest forward hit, or None on a miss.
    """
    b_m = a_m * (1.0 - f)
    sx = 1.0 / (a_m * a_m)
    sy = sx
    sz = 1.0 / (b_m * b_m)
    d0, d1, d2 = float(d_ecef[0]), float(d_ecef[1]), float(d_ecef[2])
    r0, r1, r2 = float(r0_ecef_m[0]), float(r0_ecef_m[1]), float(r0_ecef_m[2])
    quad_a = sx * d0 * d0 + sy * d1 * d1 + sz * d2 * d2
    quad_b = 2.0 * (sx * r0 * d0 + sy * r1 * d1 + sz * r2 * d2)
    quad_c = sx * r0 * r0 + sy * r1 * r1 + sz * r2 * r2 - 1.0
    if abs(quad_a) < 1e-30:
        return None
    disc = quad_b * quad_b - 4.0 * quad_a * quad_c
    if disc < 0.0:
        return None
    sqrt_disc = math.sqrt(disc)
    t1 = (-quad_b - sqrt_disc) / (2.0 * quad_a)
    t2 = (-quad_b + sqrt_disc) / (2.0 * quad_a)
    hits = [t for t in (t1, t2) if t > _MIN_RANGE_M]
    if not hits:
        return None
    t_hit = min(hits)
    hit = r0_ecef_m + t_hit * d_ecef
    return hit, t_hit
