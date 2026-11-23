"""TRACKING node equivalence: rebase, residual reset, and hunt interplay."""

import math
from dataclasses import replace

import numpy as np
from flight.libs.config import EphemerisConfig, SensorConfig
from flight.libs.types import ActivationKey
from flight.payload.gimbal.intersect import CameraGeometry, intersect_cog
from flight.payload.gimbal.predictor import predict_los
from flight.payload.gimbal.request import InhibitReference
from flight.payload.graphs import operate
from flight.payload.graphs.operate import tracking
from flight.payload.graphs.parameters import GraphParameters
from flight.payload.records import (
    IssSample,
)

from .support import BlobBuilder, TickBuilder, VisionBuilder

_SENSOR = SensorConfig()
_BORESIGHT = (_SENSOR.width_px / 2.0, _SENSOR.height_px / 2.0)


def _iss(dt_s: float = 0.0) -> IssSample:
    """Circular-LEO IssSample `dt_s` after the ephemeris epoch."""
    eph = EphemerisConfig()
    radius = 6_378_137.0 + 400_000.0
    speed = math.sqrt(eph.mu_m3_s2 / radius)
    theta = (speed / radius) * dt_s
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return IssSample(
        r_m=(radius * cos_t, radius * sin_t, 0.0),
        v_m_s=(-speed * sin_t, speed * cos_t, 0.0),
        utc_s=eph.epoch_utc_s + dt_s,
    )


def _camera(params: GraphParameters) -> CameraGeometry:
    """Band-plane pinhole geometry from the parameters."""
    return params.camera


def _intersect(
    params: GraphParameters,
    p_cog: tuple[float, float],
    theta_g_rad: float,
    iss: IssSample,
) -> tuple[float, float, float]:
    """Height-proxy CoG intersect using the configured pinhole geometry."""
    eph = params.config.ephemeris
    result = intersect_cog(
        p_cog,
        theta_g_rad,
        iss.r_m,
        iss.v_m_s,
        iss.utc_s,
        eph.epoch_utc_s,
        eph.omega_earth_rad_s,
        eph.wgs84_a_m,
        eph.wgs84_f,
        _camera(params),
        params.config.controller.predictor.cog_height_m,
    )
    assert result is not None
    return result.point_ecef_m


def _predict(
    params: GraphParameters,
    iss: IssSample,
    r_cog_ecef_m: tuple[float, float, float],
) -> float:
    """Elevation rate of a frozen ECEF CoG at one ISS sample."""
    eph = params.config.ephemeris
    los = predict_los(
        iss.utc_s,
        iss.r_m,
        iss.v_m_s,
        r_cog_ecef_m,
        eph.omega_earth_rad_s,
        eph.epoch_utc_s,
    )
    return los.elevation_rate_rad_s


def test_cog_jump_rebases_against_old_cog(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
) -> None:
    """An IoU-matched CoG jump rebases omega_res against the old CoG."""
    theta_g = math.radians(35.0)
    iss0 = _iss(0.0)
    iss1 = _iss(10.0)
    p_cog = (_BORESIGHT[0], _BORESIGHT[1])
    p1 = _intersect(params, p_cog, theta_g, iss0)
    p2 = _intersect(params, p_cog, theta_g, iss1)
    assert p2 != p1
    omega_old0 = _predict(params, iss0, p1)
    omega_old1 = _predict(params, iss1, p1)
    omega_new1 = _predict(params, iss1, p2)
    assert abs(omega_old1 - omega_old0) > 1e-8

    prior = blob(7, p_cog, bbox=(100, 100, 150, 150))
    state = replace(
        operate.initial_state(tick(0.0, key, encoder_angle_rad=theta_g), params),
        tracked_blobs=(prior,),
        aggregate_live=True,
        last_observation_s=0.0,
        target=replace(
            operate.initial_state(tick(0.0, key), params).target,
            r_cog_ecef_m=p1,
            last_omega_t_nom=omega_old0,
        ),
    )
    sample = vision(10.0, key, blobs=(blob(7, p_cog),), theta_g_rad=theta_g, iss=iss1)
    new_state, _ = operate.step(
        state, tick(10.0, key, encoder_angle_rad=theta_g, vision=sample, navigation=iss1), params
    )
    changes = new_state.residual_history.reference_changes
    assert len(changes) == 1
    assert abs(changes[0].old_rate_rad_s - omega_old1) < 1e-12
    assert abs(changes[0].old_rate_rad_s - omega_old0) > 1e-8
    assert abs(changes[0].new_rate_rad_s - omega_new1) < 1e-12
    assert new_state.target.r_cog_ecef_m == p2


def test_unmatched_blob_resets_residual(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
) -> None:
    """A tracked set with no overlapping blob_id cold-starts the residual."""
    theta_g = math.radians(20.0)
    state = operate.initial_state(tick(0.0, key), params)
    first = vision(0.0, key, blobs=(blob(1, (612.0, 442.0)),), theta_g_rad=theta_g)
    state, _ = operate.step(state, tick(0.02, key, encoder_angle_rad=theta_g, vision=first), params)
    planted = np.array([0.15, 0.04], dtype=np.float64)
    state = replace(state, residual=replace(state.residual, x=planted, has_measurement=True))
    disjoint = vision(
        0.04, key, blobs=(blob(9, (700.0, 200.0), bbox=(600, 600, 650, 650)),), theta_g_rad=theta_g
    )
    new_state, _ = operate.step(
        state, tick(0.06, key, encoder_angle_rad=theta_g, vision=disjoint), params
    )
    assert new_state.node is operate.OperateNode.TRACKING
    assert not np.allclose(new_state.residual.x, planted)


def test_single_miss_keeps_residual_and_cog(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
) -> None:
    """One empty frame while TRACKING keeps the residual and the CoG."""
    theta_g = math.radians(20.0)
    iss = _iss()
    state = operate.initial_state(tick(0.0, key), params)
    first = vision(0.0, key, blobs=(blob(1),), theta_g_rad=theta_g, iss=iss)
    state, _ = operate.step(
        state,
        tick(0.02, key, encoder_angle_rad=theta_g, vision=first, navigation=iss),
        params,
    )
    assert state.residual.has_measurement is True
    assert state.target.r_cog_ecef_m is not None
    checkpoint = state.residual_history.checkpoint.t_s
    miss = vision(0.04, key, blobs=(), theta_g_rad=theta_g, iss=iss)
    new_state, _ = operate.step(
        state,
        tick(0.06, key, encoder_angle_rad=theta_g, vision=miss, navigation=iss),
        params,
    )
    assert new_state.node is operate.OperateNode.TRACKING
    assert new_state.aggregate_live is True
    assert new_state.residual.has_measurement is True
    assert new_state.target.r_cog_ecef_m == state.target.r_cog_ecef_m
    assert new_state.residual_history.checkpoint.t_s == checkpoint


def test_explicit_reference_change_is_submitted(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
) -> None:
    """An explicit predictor-reference change is retained in the history."""
    from flight.payload.tracking import PredictorReferenceChange

    theta_g = math.radians(20.0)
    state = operate.initial_state(tick(0.0, key, encoder_angle_rad=theta_g), params)
    sample = vision(0.0, key, blobs=(blob(1),), theta_g_rad=theta_g, iss=_iss())
    state, _ = operate.step(
        state,
        tick(0.02, key, encoder_angle_rad=theta_g, vision=sample, navigation=_iss()),
        params,
    )
    change = PredictorReferenceChange(
        change_id="ref:explicit",
        t_s=0.04,
        old_rate_rad_s=0.01,
        new_rate_rad_s=0.02,
    )
    new_state, _ = operate.step(
        state,
        tick(0.06, key, encoder_angle_rad=theta_g, navigation=_iss(), reference_change=change),
        params,
    )
    ids = [item.change_id for item in new_state.residual_history.reference_changes]
    assert "ref:explicit" in ids


def test_delayed_vision_replay_records_accepted_disposition(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
) -> None:
    """A delayed sample bracketed by encoder history replays as ACCEPTED."""
    from flight.payload.tracking.residual import ObservationDisposition

    theta_g = math.radians(20.0)
    state = operate.initial_state(tick(0.0, key, encoder_angle_rad=theta_g), params)
    state, _ = operate.step(state, tick(0.02, key, encoder_angle_rad=theta_g), params)
    sample = vision(0.05, key, blobs=(blob(1),), theta_g_rad=theta_g, iss=_iss())
    new_state, _ = operate.step(
        state,
        tick(0.06, key, encoder_angle_rad=theta_g, vision=sample, navigation=_iss()),
        params,
    )
    assert new_state.vision_disposition is ObservationDisposition.ACCEPTED
    assert new_state.last_e_az != 0.0 or new_state.target.r_cog_ecef_m is not None


def test_tracking_node_directly_inhibits_on_stale(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """The TRACKING node step inhibits without fresh encoder feedback."""
    state = operate.initial_state(tick(0.0, key), params)
    inputs = tick(1.0, key, encoder_t_s=0.0)
    new_state, outcome = tracking.step(state, inputs, params)
    assert new_state is state
    assert isinstance(outcome.reference, InhibitReference)


def test_hold_node_directly_captures_target(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """The HOLD node captures a target from fresh feedback when unset."""
    from flight.payload.gimbal.request import PoseReference
    from flight.payload.graphs.operate import hold

    state = operate.State(
        activation_key=key,
        node=operate.OperateNode.HOLD,
        tracked_blobs=(),
        aggregate_live=False,
        last_observation_s=None,
        miss_count=0,
        loss_handled=False,
        rewind_entered_s=None,
        residual=params.residual_filter.initial_state(),
        residual_history=params.residual_filter.initial_history(),
        target=operate.initial_state(tick(0.0, key), params).target,
        hold=operate.initial_state(tick(0.0, key), params).hold,
        seen_vision=(),
    )
    new_state, outcome = hold.step(
        state, tick(0.02, key, encoder_angle_rad=math.radians(30.0)), params
    )
    assert new_state.hold.target_rad == math.radians(30.0)
    assert isinstance(outcome.reference, PoseReference)
