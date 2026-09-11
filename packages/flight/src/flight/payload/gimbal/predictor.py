"""Co-rotating CoG elevation and optical-azimuth predictor (pure).

Given ISS ECI state and a frozen ECEF CoG, returns a LosPrediction with signed
off-nadir elevation, co-rotating elevation rate, and unactuated optical
azimuth rate. Does not finite-difference successive intersects.

omega_el includes Earth rotation in the actuated elevation axis; it is not
orbital mean motion. omega_az is the unactuated lateral rate of
atan2(ly, hypot(lx, lz)) (equator cross-track Earth rotation at nadir). omega_az
is never commanded. LVLH y_hat is treated as inertially fixed (y_dot = 0).

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

# stdlib
import math
from dataclasses import dataclass

# third-party
import numpy as np

# internal
from flight.payload.gimbal.geo import eci_from_ecef, lvlh_axes


@dataclass(frozen=True, slots=True)
class LosPrediction:
    """Elevation and dual-rate prediction of a frozen ECEF CoG.

    Attributes:
        elevation_rad: Signed off-nadir elevation, atan2(lx, lz).
        elevation_rate_rad_s: Analytic omega_el, including Earth rotation.
        azimuth_rate_rad_s: Unactuated optical-azimuth rate omega_az.

    Notes:
        LVLH y_hat is treated as inertially fixed (y_dot = 0). Callers read
        named fields. They do not unpack interchangeable floats.
    """

    elevation_rad: float
    elevation_rate_rad_s: float
    azimuth_rate_rad_s: float


def predict_los(
    utc_s: float,
    r_iss_eci_m: tuple[float, float, float],
    v_iss_eci_m_s: tuple[float, float, float],
    r_cog_ecef_m: tuple[float, float, float],
    omega_earth_rad_s: float,
    epoch_utc_s: float,
) -> LosPrediction:
    """Elevation, elevation rate, and unactuated optical-azimuth rate of a frozen ECEF CoG.

    Inputs:
        utc_s: UTC seconds for Earth rotation.
        r_iss_eci_m, v_iss_eci_m_s: ISS ECI state (meters, m/s).
        r_cog_ecef_m: CoG Earth point in ECEF meters (held fixed).
        omega_earth_rad_s: Earth rotation rate.
        epoch_utc_s: UTC seconds at which ECEF and ECI axes coincide.

    Outputs:
        LosPrediction: Named elevation, elevation rate, and azimuth rate.

    Notes:
        omega_el is the analytic Jacobian of atan2(lx, lz) with
        look_dot = Omega_E x r_cog_eci - v_iss plus LVLH x/z rates. It includes
        Earth rotation in the actuated elevation axis; it is not orbital mean
        motion. omega_az is d/dt of atan2(ly, hypot(lx, lz)), the optical azimuth
        off the elevation plane. It is unactuated lateral rate (equator
        cross-track Earth rotation at nadir) and is never commanded. LVLH y_hat
        is treated as inertially fixed (y_dot = 0). A degenerate ISS range
        returns elevation with both rates 0.0.
    """
    r_iss = np.asarray(r_iss_eci_m, dtype=np.float64)  # np.ndarray[float64, (3,)]
    v_iss = np.asarray(v_iss_eci_m_s, dtype=np.float64)  # np.ndarray[float64, (3,)]
    r_cog_ecef = np.asarray(r_cog_ecef_m, dtype=np.float64)  # np.ndarray[float64, (3,)]
    r_cog_eci = eci_from_ecef(r_cog_ecef, omega_earth_rad_s, utc_s, epoch_utc_s)
    look = r_cog_eci - r_iss  # np.ndarray[float64, (3,)]
    x_hat, y_hat, z_hat = lvlh_axes(r_iss, v_iss)
    lx = float(look @ x_hat)
    ly = float(look @ y_hat)
    lz = float(look @ z_hat)
    theta = math.atan2(lx, lz)

    r_norm = float(np.linalg.norm(r_iss))
    if r_norm < 1.0:
        return LosPrediction(
            elevation_rad=theta,
            elevation_rate_rad_s=0.0,
            azimuth_rate_rad_s=0.0,
        )

    omega_e = np.array([0.0, 0.0, omega_earth_rad_s], dtype=np.float64)  # np.ndarray[float64, (3,)]
    r_cog_dot = np.cross(omega_e, r_cog_eci)  # np.ndarray[float64, (3,)]
    look_dot = r_cog_dot - v_iss  # np.ndarray[float64, (3,)]

    r_dot_v = float(r_iss @ v_iss)
    z_dot = -(v_iss / r_norm - r_iss * (r_dot_v / r_norm**3))  # np.ndarray[float64, (3,)]
    y_dot = np.zeros(3, dtype=np.float64)  # np.ndarray[float64, (3,)]
    x_dot = np.cross(y_dot, z_hat) + np.cross(y_hat, z_dot)  # np.ndarray[float64, (3,)]

    dlx = float(look_dot @ x_hat + look @ x_dot)
    dly = float(look_dot @ y_hat + look @ y_dot)
    dlz = float(look_dot @ z_hat + look @ z_dot)
    denom_el = lx * lx + lz * lz
    omega_el = (lz * dlx - lx * dlz) / denom_el if denom_el > 1e-12 else 0.0

    rho = math.hypot(lx, lz)
    denom_az = rho * rho + ly * ly
    if rho <= 1e-12 or denom_az <= 1e-12:
        omega_az = 0.0
    else:
        drho = (lx * dlx + lz * dlz) / rho
        omega_az = (rho * dly - ly * drho) / denom_az
    return LosPrediction(
        elevation_rad=theta,
        elevation_rate_rad_s=float(omega_el),
        azimuth_rate_rad_s=float(omega_az),
    )
