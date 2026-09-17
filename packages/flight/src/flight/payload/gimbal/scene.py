"""Scene-point selection and residual-reference identity (pure).

TRACKING predicts from the stored CoG Earth point. REWIND predicts from the
boresight intersect with the 2 km height-proxy ellipsoid. That boresight hit is
a smear-limited scene rate only. It is not a target CoG. SAFE has no scene.

Prediction time is ISS UTC when navigation is present. It is not outer monotonic
now. Missing ISS is nav_valid False with los None. A physically stationary scene
still carries a LosPrediction whose rates may be 0.0.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001, REQ-GIMB-HIGH-003.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from flight.libs.types import GimbalState
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


def acquire_resets_residual(
    *,
    previous_mode: GimbalState,
    new_mode: GimbalState,
    previous_aggregate_live: bool,
    previous_blob_ids: frozenset[int],
    new_blob_ids: frozenset[int],
) -> bool:
    """True when TRACKING has acquired a target that needs a cold residual.

    Inputs:
        previous_mode: Arbiter mode before this outer tick.
        new_mode: Arbiter mode after this outer tick.
        previous_aggregate_live: Aggregate liveness before this outer tick.
        previous_blob_ids: blob_id values on state.arbiter.tracked_blobs before
            arbiter.step. Those IDs already include ingest_inference IoU match.
        new_blob_ids: blob_id values on this tick's vision.blobs. Empty when
            this tick has no vision sample or no accepted aggregate.

    Outputs:
        bool: True when the residual filter must cold-start.

    Notes:
        A reset is an acquire, not a loss. Cases:
        1. First blob from cold (no prior live aggregate).
        2. Blob that enters TRACKING from REWIND (the hunt moved the gimbal).
        3. TRACKING blob set with no overlapping blob_id versus the previous
           tracked set (new object). An empty previous set is not this case;
           a single miss that cleared tracked_blobs still coasts.
        A single empty frame while still TRACKING does not reset.
    """
    if new_mode is not GimbalState.TRACKING:
        return False
    if not new_blob_ids:
        return False
    if previous_mode is GimbalState.REWIND:
        return True
    if not previous_aggregate_live:
        return True
    return bool(previous_blob_ids) and previous_blob_ids.isdisjoint(new_blob_ids)


def select_scene(
    mode: GimbalState,
    r_cog_ecef_m: tuple[float, float, float] | None,
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
    """Select the scene Earth point and predict co-rotating rates.

    Inputs:
        mode: Arbiter mode after this outer tick.
        r_cog_ecef_m: Stored target CoG, ECEF meters, or None. Ignored in
            REWIND and SAFE.
        r_iss_eci_m, v_iss_eci_m_s, utc_s: ISS ECI state and UTC, or None.
            All three must be present for navigation to be valid.
        theta_g_rad: Encoder elevation, rad. Used for the REWIND boresight ray.
        height_m: Height-proxy ellipsoid offset, meters.
        omega_earth_rad_s, epoch_utc_s: Earth rotation and ECI/ECEF epoch.
        wgs84_a_m, wgs84_f: Surface ellipsoid scalars.

    Outputs:
        SceneEstimate: Source, optional ECEF point, optional LosPrediction, and
            navigation validity. Prediction time is ISS UTC when nav_valid.

    Notes:
        REWIND uses boresight intersect as scene rate only. The hit is not a
        CoG. TRACKING uses the stored CoG. Unknown navigation leaves los None.
        Do not read a missing scene as omega = 0.0.
    """
    nav_valid = r_iss_eci_m is not None and v_iss_eci_m_s is not None and utc_s is not None
    t_utc_s = utc_s if nav_valid else None

    if mode is GimbalState.SAFE:
        return SceneEstimate(
            t_utc_s=t_utc_s,
            source=SceneSource.NONE,
            point_ecef_m=None,
            los=None,
            nav_valid=nav_valid,
        )

    source = SceneSource.NONE
    point_ecef_m: tuple[float, float, float] | None = None
    if mode is GimbalState.REWIND:
        source = SceneSource.BORESIGHT
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
    elif r_cog_ecef_m is not None:
        source = SceneSource.COG
        point_ecef_m = r_cog_ecef_m

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
