"""Co-rotating CoG elevation predictor (pure).

Given ISS ECI state and a frozen ECEF CoG, returns signed off-nadir elevation of
the line of sight and the co-rotating elevation rate. Does not finite-difference
successive intersects.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

# stdlib
import math

# third-party
import numpy as np

# internal
from flight.payload.gimbal.geo import eci_from_ecef, lvlh_axes


def predict_los(
    utc_s: float,
    r_iss_eci_m: tuple[float, float, float],
    v_iss_eci_m_s: tuple[float, float, float],
    r_cog_ecef_m: tuple[float, float, float],
    omega_earth_rad_s: float,
    epoch_utc_s: float,
) -> tuple[float, float]:
    """Elevation and co-rotating elevation rate of a frozen ECEF CoG.

    Inputs:
        utc_s: UTC seconds for Earth rotation.
        r_iss_eci_m, v_iss_eci_m_s: ISS ECI state (meters, m/s).
        r_cog_ecef_m: CoG Earth point in ECEF meters (held fixed).
        omega_earth_rad_s: Earth rotation rate.
        epoch_utc_s: UTC seconds at which ECEF and ECI axes coincide.

    Outputs:
        tuple[float, float]: (theta_los_rad, omega_t_nom_rad_s).
    """
    r_iss = np.asarray(r_iss_eci_m, dtype=np.float64)
    v_iss = np.asarray(v_iss_eci_m_s, dtype=np.float64)
    r_cog_ecef = np.asarray(r_cog_ecef_m, dtype=np.float64)
    r_cog_eci = eci_from_ecef(r_cog_ecef, omega_earth_rad_s, utc_s, epoch_utc_s)
    look = r_cog_eci - r_iss
    x_hat, y_hat, z_hat = lvlh_axes(r_iss, v_iss)
    ly = float(look @ y_hat)
    lz = float(look @ z_hat)
    theta = math.atan2(ly, lz)

    omega_e = np.array([0.0, 0.0, omega_earth_rad_s], dtype=np.float64)
    r_cog_dot = np.cross(omega_e, r_cog_eci)
    look_dot = r_cog_dot - v_iss

    r_norm = float(np.linalg.norm(r_iss))
    if r_norm < 1.0:
        return theta, 0.0
    r_dot_v = float(r_iss @ v_iss)
    z_dot = -(v_iss / r_norm - r_iss * (r_dot_v / r_norm**3))
    x_dot = np.zeros(3, dtype=np.float64)
    y_dot = np.cross(z_dot, x_hat) + np.cross(z_hat, x_dot)

    dly = float(look_dot @ y_hat + look @ y_dot)
    dlz = float(look_dot @ z_hat + look @ z_dot)
    denom = ly * ly + lz * lz
    omega_nom = (lz * dly - ly * dlz) / denom if denom > 1e-12 else 0.0
    return theta, float(omega_nom)
