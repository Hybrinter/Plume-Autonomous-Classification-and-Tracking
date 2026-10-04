"""Scene-point prediction entry points: stored CoG or boresight intersect (pure).

cog_scene predicts from the stored CoG Earth point. boresight_scene predicts
from the boresight intersect with the 2 km height-proxy ellipsoid. That
boresight hit is a smear-limited scene rate only. It is not a target CoG. The
caller selects which entry point applies; there is no mode dispatch here.

Prediction time is ISS UTC when navigation is present. It is not outer monotonic
now. Missing ISS is nav_valid False with los None. A physically stationary scene
still carries a LosPrediction whose rates may be 0.0.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001, REQ-GIMB-HIGH-003.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from flight.payload.gimbal.intersect import intersect_boresight
from flight.payload.gimbal.predictor import LosPrediction, predict_los


class SceneSource(Enum):
    """Earth-point source selected for one outer tick.

    String values mirror member names.
    """

    COG = "COG"
    BORESIGHT = "BORESIGHT"
    NONE = "NONE"


@dataclass(frozen=True, slots=True)
class SceneEstimate:
    """One scene-point selection and its co-rotating prediction.

    Attributes:
        t_utc_s: ISS UTC seconds used for Earth rotation, or None when
            navigation is unknown. This is not outer monotonic now.
        source: Selected Earth-point kind.
        point_ecef_m: Selected ECEF point, or None when no point exists.
        los: Co-rotating prediction, or None when the predictor did not run.
        nav_valid: True when ISS ECI state and UTC were supplied. False is
            unknown navigation, not a zero-rate scene.
    """

    t_utc_s: float | None
    source: SceneSource
    point_ecef_m: tuple[float, float, float] | None
    los: LosPrediction | None
    nav_valid: bool


def _predict_scene(
    source: SceneSource,
    point_ecef_m: tuple[float, float, float] | None,
    r_iss_eci_m: tuple[float, float, float] | None,
    v_iss_eci_m_s: tuple[float, float, float] | None,
    utc_s: float | None,
    omega_earth_rad_s: float,
    epoch_utc_s: float,
) -> SceneEstimate:
    """Build a SceneEstimate: predict los only with full nav and a point."""
    nav_valid = r_iss_eci_m is not None and v_iss_eci_m_s is not None and utc_s is not None
    t_utc_s = utc_s if nav_valid else None
    los: LosPrediction | None = None
    if (
        r_iss_eci_m is not None
        and v_iss_eci_m_s is not None
        and utc_s is not None
        and point_ecef_m is not None
    ):
        los = predict_los(
            utc_s,
            r_iss_eci_m,
            v_iss_eci_m_s,
            point_ecef_m,
            omega_earth_rad_s,
            epoch_utc_s,
        )
    return SceneEstimate(
        t_utc_s=t_utc_s,
        source=source,
        point_ecef_m=point_ecef_m,
        los=los,
        nav_valid=nav_valid,
    )


def cog_scene(
    r_cog_ecef_m: tuple[float, float, float] | None,
    r_iss_eci_m: tuple[float, float, float] | None,
    v_iss_eci_m_s: tuple[float, float, float] | None,
    utc_s: float | None,
    omega_earth_rad_s: float,
    epoch_utc_s: float,
) -> SceneEstimate:
    """Predict the scene rate from the stored target CoG Earth point.

    Inputs:
        r_cog_ecef_m: Stored target CoG, ECEF meters, or None.
        r_iss_eci_m, v_iss_eci_m_s, utc_s: ISS ECI state and UTC, or None.
            All three must be present for navigation to be valid.
        omega_earth_rad_s, epoch_utc_s: Earth rotation and ECI/ECEF epoch.

    Outputs:
        SceneEstimate: Source COG and the stored point when a CoG exists, source
            NONE otherwise. los is None when navigation is unknown; a missing
            scene is not omega = 0.0.
    """
    source = SceneSource.COG if r_cog_ecef_m is not None else SceneSource.NONE
    return _predict_scene(
        source,
        r_cog_ecef_m,
        r_iss_eci_m,
        v_iss_eci_m_s,
        utc_s,
        omega_earth_rad_s,
        epoch_utc_s,
    )


def boresight_scene(
    r_iss_eci_m: tuple[float, float, float] | None,
    v_iss_eci_m_s: tuple[float, float, float] | None,
    utc_s: float | None,
    theta_g_rad: float,
    height_m: float,
    omega_earth_rad_s: float,
    epoch_utc_s: float,
    wgs84_a_m: float,
    wgs84_f: float,
) -> SceneEstimate:
    """Predict the hunt scene rate from the boresight height-proxy intersect.

    Inputs:
        r_iss_eci_m, v_iss_eci_m_s, utc_s: ISS ECI state and UTC, or None.
            All three must be present for navigation to be valid.
        theta_g_rad: Encoder elevation, rad. Used for the hunt boresight ray.
        height_m: Height-proxy ellipsoid offset, meters.
        omega_earth_rad_s, epoch_utc_s: Earth rotation and ECI/ECEF epoch.
        wgs84_a_m, wgs84_f: Surface ellipsoid scalars.

    Outputs:
        SceneEstimate: Source BORESIGHT even when navigation is absent or the
            ray misses the ellipsoid; point_ecef_m is the hit or None. The hit
            is scene rate only; do not store it as a target CoG.
    """
    point_ecef_m: tuple[float, float, float] | None = None
    if r_iss_eci_m is not None and v_iss_eci_m_s is not None and utc_s is not None:
        bore = intersect_boresight(
            theta_g_rad,
            r_iss_eci_m,
            v_iss_eci_m_s,
            utc_s,
            epoch_utc_s,
            omega_earth_rad_s,
            wgs84_a_m,
            wgs84_f,
            height_m,
        )
        if bore is not None:
            point_ecef_m = bore.point_ecef_m
    return _predict_scene(
        SceneSource.BORESIGHT,
        point_ecef_m,
        r_iss_eci_m,
        v_iss_eci_m_s,
        utc_s,
        omega_earth_rad_s,
        epoch_utc_s,
    )
