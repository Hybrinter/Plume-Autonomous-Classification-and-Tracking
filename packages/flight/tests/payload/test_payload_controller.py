"""Tests for the PayloadController cascaded inner/outer cores."""

import math
from dataclasses import replace

import numpy as np
from flight.libs.config import ControllerConfig, EphemerisConfig, GimbalConfig, SensorConfig
from flight.libs.messages import BlobMeta, InferenceResultMsg
from flight.libs.types import GimbalCommandMode, GimbalState, MessageType
from flight.payload.control import IssSample, PayloadController, VisionSample
from flight.payload.gimbal.arbiter import ArbiterState
from flight.payload.gimbal.intersect import CameraGeometry, intersect_cog
from flight.payload.gimbal.predictor import predict_los
from flight.payload.tracking import EncoderSample

_BORESIGHT_X = 612.0
_BORESIGHT_Y = 512.0


def _encoder(t_s: float, angle_rad: float = 0.0) -> EncoderSample:
    """Build a valid timestamped encoder sample for one outer tick."""
    return EncoderSample(f"encoder:{t_s:.6f}", t_s, angle_rad, 0.0)


def _controller() -> PayloadController:
    """Build a controller with default controller, sensor, and gimbal geometry."""
    return PayloadController.from_config(
        ControllerConfig(), SensorConfig(), GimbalConfig(), EphemerisConfig()
    )


def _result(
    frame_id: int,
    *,
    centroid: tuple[float, float] | None,
    bbox: tuple[int, int, int, int] = (100, 100, 150, 150),
) -> InferenceResultMsg:
    """Build an InferenceResultMsg, optionally carrying one strong blob at `centroid`."""
    blobs: tuple[BlobMeta, ...] = ()
    if centroid is not None:
        blobs = (
            BlobMeta(
                blob_id=1,
                bbox=bbox,
                centroid_raw=centroid,
                pixel_area=200,
                mean_confidence=0.85,
                persistence_count=1,
            ),
        )
    return InferenceResultMsg(
        msg_type=MessageType.INFERENCE_RESULT,
        timestamp_utc="2026-06-01T00:00:00.000Z",
        frame_id=frame_id,
        mask=np.zeros((16, 16), dtype=np.float32),
        blobs=blobs,
        model_version="test",
        inference_ms=0.0,
        mode_flags=0,
    )


def _blob(blob_id: int, centroid: tuple[float, float], pixel_area: int) -> BlobMeta:
    """Build one accepted connected component for aggregate-centroid tests."""
    return BlobMeta(
        blob_id=blob_id,
        bbox=(100, 100, 150, 150),
        centroid_raw=centroid,
        pixel_area=pixel_area,
        mean_confidence=0.85,
        persistence_count=1,
    )


def test_initial_state_is_tracking_cold() -> None:
    """The controller starts TRACKING with r=0 and no residual measurement."""
    state = _controller().initial_state()
    assert state.arbiter.gimbal_state is GimbalState.TRACKING
    assert state.arbiter.tracked_blobs == ()
    assert state.r_rad_s == 0.0
    assert state.residual.has_measurement is False


def test_cold_outer_holds_r_zero_without_vision() -> None:
    """Coast ticks before the first blob keep r = 0."""
    controller = _controller()
    state = controller.initial_state()
    tick = controller.outer_step(state, 0.02, _encoder(0.02), None, None, False, False)
    assert tick.state.arbiter.gimbal_state is GimbalState.TRACKING
    assert tick.state.r_rad_s == 0.0
    assert tick.request is None
    assert tick.fault is None


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


def _camera(controller: PayloadController) -> CameraGeometry:
    """Band-plane pinhole geometry from the controller."""
    return CameraGeometry(
        width_px=controller.plane_width_px,
        height_px=controller.plane_height_px,
        pixel_pitch_m=controller.pixel_pitch_m,
        focal_length_m=controller.focal_m,
    )


def _predict(
    controller: PayloadController,
    iss: IssSample,
    r_cog_ecef_m: tuple[float, float, float],
) -> float:
    """Elevation rate of a frozen ECEF CoG at one ISS sample."""
    los = predict_los(
        iss.utc_s,
        iss.r_m,
        iss.v_m_s,
        r_cog_ecef_m,
        controller.eph.omega_earth_rad_s,
        controller.eph.epoch_utc_s,
    )
    return los.elevation_rate_rad_s


def _intersect(
    controller: PayloadController,
    p_cog: tuple[float, float],
    theta_g_rad: float,
    iss: IssSample,
) -> tuple[float, float, float]:
    """Height-proxy CoG intersect using the controller pinhole geometry."""
    result = intersect_cog(
        p_cog,
        theta_g_rad,
        iss.r_m,
        iss.v_m_s,
        iss.utc_s,
        controller.eph.epoch_utc_s,
        controller.eph.omega_earth_rad_s,
        controller.eph.wgs84_a_m,
        controller.eph.wgs84_f,
        _camera(controller),
        controller.cfg.predictor.cog_height_m,
    )
    assert result is not None
    return result.point_ecef_m


def test_blob_above_boresight_commands_positive_r() -> None:
    """An above-boresight blob snaps e and produces a positive elevation rate."""
    controller = _controller()
    state = controller.initial_state()
    centroid = (_BORESIGHT_X, _BORESIGHT_Y - 70.0)
    iss = _iss()
    state, sample = controller.ingest_inference(
        state, _result(1, centroid=centroid), 0.0, 1000.0, iss
    )
    assert sample.z_v is not None
    assert sample.z_v > 0.0
    tick = controller.outer_step(state, 0.02, _encoder(0.0), sample, iss, False, False)
    assert tick.state.arbiter.gimbal_state is GimbalState.TRACKING
    assert tick.state.residual.has_measurement is True
    assert tick.state.r_rad_s > 0.0
    assert tick.request is None


def test_ingest_uses_area_weighted_aggregate_centroid() -> None:
    """Every accepted component contributes to the visible union centroid."""
    controller = _controller()
    result = InferenceResultMsg(
        msg_type=MessageType.INFERENCE_RESULT,
        timestamp_utc="2026-06-01T00:00:00.000Z",
        frame_id=1,
        mask=np.zeros((16, 16), dtype=np.float32),
        blobs=(
            _blob(1, (100.0, 300.0), 100),
            _blob(2, (700.0, 700.0), 300),
        ),
        model_version="test",
        inference_ms=0.0,
        mode_flags=0,
    )

    _state, sample = controller.ingest_inference(controller.initial_state(), result, 0.0, 1000.0)

    assert sample.p_cog == (550.0, 600.0)


def test_visual_tracking_does_not_require_navigation() -> None:
    """A valid visual aggregate commands tracking when ephemeris is unavailable."""
    controller = _controller()
    state, sample = controller.ingest_inference(
        controller.initial_state(),
        _result(1, centroid=(_BORESIGHT_X, _BORESIGHT_Y - 70.0)),
        0.0,
        1000.0,
    )

    tick = controller.outer_step(state, 0.02, _encoder(0.0), sample, None, False, False)

    assert tick.state.arbiter.aggregate_live is True
    assert tick.state.r_rad_s > 0.0


def test_aggregate_coast_expires_into_return() -> None:
    """A stalled vision pipeline makes an observed target return toward +45 degrees."""
    controller = _controller()
    state, sample = controller.ingest_inference(
        controller.initial_state(),
        _result(1, centroid=(_BORESIGHT_X, _BORESIGHT_Y - 70.0)),
        0.0,
        1000.0,
    )
    live = controller.outer_step(state, 0.02, _encoder(0.0), sample, None, False, False).state

    expired = controller.outer_step(
        live,
        controller.cfg.arbiter.max_observation_age_s + 0.04,
        _encoder(controller.cfg.arbiter.max_observation_age_s + 0.04, 0.0),
        None,
        None,
        False,
        False,
    )

    assert expired.state.arbiter.gimbal_state is GimbalState.REWIND
    assert expired.state.arbiter.aggregate_live is False
    assert expired.state.r_rad_s > 0.0


def test_safe_entry_produces_stow_request() -> None:
    """A commanded SAFE produces a STOW request and latched SAFE state."""
    controller = _controller()
    state = controller.initial_state()
    tick = controller.outer_step(state, 0.02, _encoder(0.02), None, None, True, False)
    assert tick.request is not None
    assert tick.request.mode is GimbalCommandMode.STOW
    assert tick.state.arbiter.gimbal_state is GimbalState.SAFE
    assert tick.state.pose_mode is GimbalCommandMode.STOW
    assert tick.state.r_rad_s < 0.0


def test_inner_step_writes_torque() -> None:
    """inner_step with a nonzero r produces a torque command."""
    from dataclasses import replace

    controller = _controller()
    state = replace(controller.initial_state(), r_rad_s=math.radians(1.0))
    tick = controller.inner_step(state, 0.001, 0.0)
    assert tick.tau_nm != 0.0


def test_iss_sample_feeds_predictor() -> None:
    """An IssSample with a stored CoG at the 2 km proxy produces a finite omega_t_nom."""
    controller = _controller()
    from dataclasses import replace

    eph = EphemerisConfig()
    r = 6_378_137.0 + 400_000.0
    v = math.sqrt(eph.mu_m3_s2 / r)
    iss = IssSample(r_m=(r, 0.0, 0.0), v_m_s=(0.0, v, 0.0), utc_s=eph.epoch_utc_s)
    cog = (eph.wgs84_a_m, 0.0, 0.0)
    state = replace(controller.initial_state(), r_cog_ecef_m=cog)
    sample = VisionSample(
        t_s=0.0,
        z_v=0.0,
        p_cog=(612.0, 512.0),
        exposure_us=1000.0,
        blobs=(),
        mode_flags=0,
        iss=iss,
        frame_id="1",
    )
    tick = controller.outer_step(state, 0.02, _encoder(0.0), sample, iss, False, False)
    assert math.isfinite(tick.state.last_omega_t_nom)
    assert math.isfinite(tick.state.last_omega_az_nom)


def test_rewind_uses_boresight_not_plume_cog() -> None:
    """REWIND predicts from current boresight at 2 km, not the lost-plume ECEF point."""
    from dataclasses import replace

    from flight.payload.gimbal.intersect import intersect_boresight
    from flight.payload.gimbal.predictor import predict_los

    controller = _controller()
    eph = EphemerisConfig()
    iss = _iss()
    plume = (eph.wgs84_a_m, 0.0, 0.0)
    state, sample = controller.ingest_inference(
        controller.initial_state(),
        _result(1, centroid=(_BORESIGHT_X, _BORESIGHT_Y - 70.0)),
        0.0,
        1000.0,
        iss,
    )
    live = controller.outer_step(state, 0.02, _encoder(0.0), sample, iss, False, False).state
    live = replace(live, r_cog_ecef_m=plume)
    now = controller.cfg.arbiter.max_observation_age_s + 0.04
    theta_g = math.radians(20.0)
    expired = controller.outer_step(live, now, _encoder(now, theta_g), None, iss, False, False)
    assert expired.state.arbiter.gimbal_state is GimbalState.REWIND
    assert expired.state.r_cog_ecef_m is None
    assert expired.state.residual_history.events is live.residual_history.events
    assert np.array_equal(expired.state.residual.x, live.residual.x)
    height_m = controller.cfg.predictor.cog_height_m
    bore = intersect_boresight(
        theta_g,
        iss.r_m,
        iss.v_m_s,
        iss.utc_s,
        eph.epoch_utc_s,
        eph.omega_earth_rad_s,
        eph.wgs84_a_m,
        eph.wgs84_f,
        height_m,
    )
    assert bore is not None
    omega_bore = predict_los(
        iss.utc_s,
        iss.r_m,
        iss.v_m_s,
        bore.point_ecef_m,
        eph.omega_earth_rad_s,
        eph.epoch_utc_s,
    ).elevation_rate_rad_s
    omega_plume = predict_los(
        iss.utc_s,
        iss.r_m,
        iss.v_m_s,
        plume,
        eph.omega_earth_rad_s,
        eph.epoch_utc_s,
    ).elevation_rate_rad_s
    assert abs(expired.state.last_omega_t_nom - omega_bore) < 1e-9
    assert abs(omega_bore - omega_plume) > 1e-8


def test_cog_jump_rebases_against_old_cog_at_current_iss() -> None:
    """An IoU-matched CoG jump rebases omega_res using the old CoG at this ISS sample."""
    from dataclasses import replace

    controller = _controller()
    theta_g = math.radians(35.0)
    iss0 = _iss(0.0)
    iss1 = _iss(10.0)
    p_cog = (_BORESIGHT_X, _BORESIGHT_Y)
    p1 = _intersect(controller, p_cog, theta_g, iss0)
    p2 = _intersect(controller, p_cog, theta_g, iss1)
    assert p2 != p1
    omega_old0 = _predict(controller, iss0, p1)
    omega_old1 = _predict(controller, iss1, p1)
    omega_new1 = _predict(controller, iss1, p2)
    assert abs(omega_old1 - omega_old0) > 1e-8

    blob = BlobMeta(
        blob_id=7,
        bbox=(100, 100, 150, 150),
        centroid_raw=p_cog,
        pixel_area=200,
        mean_confidence=0.85,
        persistence_count=2,
    )
    state = replace(
        controller.initial_state(),
        r_cog_ecef_m=p1,
        last_omega_t_nom=omega_old0,
        arbiter=ArbiterState(
            gimbal_state=GimbalState.TRACKING,
            tracked_blobs=(blob,),
            current_target_id=None,
            miss_count=0,
            aggregate_live=True,
            last_observation_s=0.0,
        ),
    )
    vision = VisionSample(
        t_s=10.0,
        frame_id="cog-jump",
        z_v=0.0,
        p_cog=p_cog,
        exposure_us=1000.0,
        blobs=(blob,),
        mode_flags=0,
        iss=iss1,
        theta_g_rad=theta_g,
    )
    tick = controller.outer_step(state, 10.0, _encoder(10.0, theta_g), vision, iss1, False, False)
    changes = tick.state.residual_history.reference_changes
    assert len(changes) == 1
    assert abs(changes[0].old_rate_rad_s - omega_old1) < 1e-12
    assert abs(changes[0].old_rate_rad_s - omega_old0) > 1e-8
    assert abs(changes[0].new_rate_rad_s - omega_new1) < 1e-12
    assert tick.state.r_cog_ecef_m == p2


def test_plume_during_rewind_acquires_with_new_cog_and_resets_residual() -> None:
    """A plume mid-REWIND returns to TRACKING, writes a new CoG, and cold-starts residual."""
    controller = _controller()
    iss = _iss()
    theta_g = math.radians(20.0)
    centroid = (_BORESIGHT_X, _BORESIGHT_Y - 70.0)
    state, sample = controller.ingest_inference(
        controller.initial_state(),
        _result(1, centroid=centroid),
        0.0,
        1000.0,
        iss,
        theta_g,
    )
    live = controller.outer_step(
        state, 0.02, _encoder(0.0, theta_g), sample, iss, False, False
    ).state
    planted_x = np.array([0.15, 0.04], dtype=np.float64)
    planted = replace(
        live,
        residual=replace(live.residual, x=planted_x, has_measurement=True),
    )
    now = controller.cfg.arbiter.max_observation_age_s + 0.04
    rewound = controller.outer_step(planted, now, _encoder(now, theta_g), None, iss, False, False)
    assert rewound.state.arbiter.gimbal_state is GimbalState.REWIND
    assert rewound.state.r_cog_ecef_m is None
    assert np.array_equal(rewound.state.residual.x, planted_x)

    t_acq = now + 0.02
    state, sample = controller.ingest_inference(
        rewound.state,
        _result(2, centroid=centroid),
        t_acq,
        1000.0,
        iss,
        theta_g,
    )
    tick = controller.outer_step(state, t_acq, _encoder(t_acq, theta_g), sample, iss, False, False)
    assert tick.state.arbiter.gimbal_state is GimbalState.TRACKING
    assert tick.state.r_cog_ecef_m is not None
    assert tick.state.residual.has_measurement is True
    assert not np.allclose(tick.state.residual.x, planted_x)
    assert tick.state.residual_history.checkpoint.t_s == t_acq
    assert "1" not in {obs.frame_id for obs in tick.state.residual_history.vision_observations}


def test_acquire_applies_vision_when_encoder_sample_is_stale() -> None:
    """A REWIND acquire snaps residual error when the encoder timestamp lags the shutter."""
    controller = _controller()
    iss = _iss()
    theta_g = math.radians(20.0)
    planted = replace(
        controller.initial_state(),
        arbiter=ArbiterState(
            gimbal_state=GimbalState.REWIND,
            tracked_blobs=(),
            current_target_id=None,
            miss_count=0,
            aggregate_live=False,
            last_observation_s=None,
            loss_handled=True,
            rewind_entered_s=0.0,
        ),
        residual=replace(
            controller.initial_state().residual,
            x=np.array([0.15, 0.04], dtype=np.float64),
            has_measurement=True,
        ),
        last_exposure_us=1000.0,
    )
    state, sample = controller.ingest_inference(
        planted,
        _result(3, centroid=(_BORESIGHT_X, _BORESIGHT_Y - 70.0)),
        2.0,
        1000.0,
        iss,
        theta_g,
    )
    tick = controller.outer_step(state, 2.0, _encoder(0.95, theta_g), sample, iss, False, False)
    assert tick.state.arbiter.gimbal_state is GimbalState.TRACKING
    assert tick.state.residual.has_measurement is True
    assert tick.state.r_rad_s > 0.0
    assert abs(float(tick.state.residual.x[0]) - 0.15) > 1e-6


def test_single_miss_does_not_reset_residual() -> None:
    """One empty frame while still TRACKING keeps the residual filter and CoG."""
    controller = _controller()
    iss = _iss()
    theta_g = math.radians(20.0)
    centroid = (_BORESIGHT_X, _BORESIGHT_Y - 70.0)
    state, sample = controller.ingest_inference(
        controller.initial_state(),
        _result(1, centroid=centroid),
        0.0,
        1000.0,
        iss,
        theta_g,
    )
    live = controller.outer_step(
        state, 0.02, _encoder(0.0, theta_g), sample, iss, False, False
    ).state
    assert live.arbiter.gimbal_state is GimbalState.TRACKING
    assert live.residual.has_measurement is True
    assert live.r_cog_ecef_m is not None
    live_checkpoint = live.residual_history.checkpoint.t_s

    miss = VisionSample(
        t_s=0.02,
        frame_id="miss",
        z_v=None,
        p_cog=None,
        exposure_us=1000.0,
        blobs=(),
        mode_flags=0,
        iss=iss,
        theta_g_rad=theta_g,
    )
    tick = controller.outer_step(live, 0.04, _encoder(0.02, theta_g), miss, iss, False, False)
    assert tick.state.arbiter.gimbal_state is GimbalState.TRACKING
    assert tick.state.arbiter.aggregate_live is True
    assert tick.state.residual.has_measurement is True
    assert tick.state.r_cog_ecef_m == live.r_cog_ecef_m
    assert tick.state.residual_history.checkpoint.t_s == live_checkpoint


def test_unmatched_blob_resets_residual() -> None:
    """A TRACKING blob with no bbox overlap versus the previous track cold-starts residual."""
    controller = _controller()
    iss = _iss()
    theta_g = math.radians(20.0)
    first, sample = controller.ingest_inference(
        controller.initial_state(),
        _result(1, centroid=(_BORESIGHT_X, _BORESIGHT_Y - 70.0), bbox=(100, 100, 150, 150)),
        0.0,
        1000.0,
        iss,
        theta_g,
    )
    live = controller.outer_step(
        first, 0.02, _encoder(0.0, theta_g), sample, iss, False, False
    ).state
    assert live.arbiter.tracked_blobs
    first_ids = {blob.blob_id for blob in live.arbiter.tracked_blobs}
    planted_x = np.array([0.15, 0.04], dtype=np.float64)
    planted = replace(
        live,
        residual=replace(live.residual, x=planted_x, has_measurement=True),
    )

    next_state, next_sample = controller.ingest_inference(
        planted,
        _result(2, centroid=(_BORESIGHT_X, _BORESIGHT_Y + 70.0), bbox=(400, 400, 450, 450)),
        0.02,
        1000.0,
        iss,
        theta_g,
    )
    next_ids = {blob.blob_id for blob in next_sample.blobs}
    assert first_ids.isdisjoint(next_ids)
    tick = controller.outer_step(
        next_state, 0.04, _encoder(0.02, theta_g), next_sample, iss, False, False
    )
    assert tick.state.arbiter.gimbal_state is GimbalState.TRACKING
    assert tick.state.residual_history.checkpoint.t_s == 0.02
    assert not np.allclose(tick.state.residual.x, planted_x)
    assert tick.state.r_cog_ecef_m is not None
    assert tick.state.r_cog_ecef_m != planted.r_cog_ecef_m
    assert "1" not in {obs.frame_id for obs in tick.state.residual_history.vision_observations}


def test_acquire_without_shutter_encoder_rejects_vision() -> None:
    """Acquire without a shutter encoder bracket does not invent an endpoint."""
    controller = _controller()
    iss = _iss()
    planted = replace(
        controller.initial_state(),
        arbiter=ArbiterState(
            gimbal_state=GimbalState.REWIND,
            tracked_blobs=(),
            current_target_id=None,
            miss_count=0,
            aggregate_live=False,
            last_observation_s=None,
            loss_handled=True,
            rewind_entered_s=0.0,
        ),
        residual=replace(
            controller.initial_state().residual,
            x=np.array([0.15, 0.04], dtype=np.float64),
            has_measurement=True,
        ),
        r_cog_ecef_m=(1.0, 2.0, 3.0),
        last_exposure_us=1000.0,
    )
    state, sample = controller.ingest_inference(
        planted,
        _result(3, centroid=(_BORESIGHT_X, _BORESIGHT_Y - 70.0)),
        2.0,
        1000.0,
        iss,
        None,
    )
    assert sample.theta_g_rad is None
    assert sample.z_v is not None
    tick = controller.outer_step(
        state,
        2.0,
        _encoder(0.95, math.radians(25.0)),
        sample,
        iss,
        False,
        False,
        timestamp_utc="2026-06-01T00:00:00.000Z",
    )
    assert tick.state.arbiter.gimbal_state is GimbalState.TRACKING
    assert tick.state.residual.has_measurement is False
    assert tick.state.residual_history.checkpoint.t_s == 2.0
    assert tick.state.residual_history.checkpoint.encoder_angle_rad is None
    assert tick.state.r_cog_ecef_m is None
    pointing = [event for event in tick.telemetry if event.event_name == "pointing"]
    assert pointing
    assert pointing[0].payload["vision_disposition"] == "no_encoder_bracket"


def test_unmatched_blob_without_intersect_drops_prior_cog() -> None:
    """Disjoint TRACKING blobs drop the old CoG when this frame has no shutter intersect."""
    controller = _controller()
    iss = _iss()
    theta_g = math.radians(20.0)
    first, sample = controller.ingest_inference(
        controller.initial_state(),
        _result(1, centroid=(_BORESIGHT_X, _BORESIGHT_Y - 70.0), bbox=(100, 100, 150, 150)),
        0.0,
        1000.0,
        iss,
        theta_g,
    )
    live = controller.outer_step(
        first, 0.02, _encoder(0.0, theta_g), sample, iss, False, False
    ).state
    assert live.r_cog_ecef_m is not None
    planted_x = np.array([0.15, 0.04], dtype=np.float64)
    planted = replace(
        live,
        residual=replace(live.residual, x=planted_x, has_measurement=True),
    )
    next_state, next_sample = controller.ingest_inference(
        planted,
        _result(2, centroid=(_BORESIGHT_X, _BORESIGHT_Y + 70.0), bbox=(400, 400, 450, 450)),
        0.02,
        1000.0,
        iss,
        None,
    )
    tick = controller.outer_step(
        next_state, 0.04, _encoder(0.02, theta_g), next_sample, iss, False, False
    )
    assert tick.state.arbiter.gimbal_state is GimbalState.TRACKING
    assert tick.state.r_cog_ecef_m is None
    assert tick.state.last_omega_t_nom == 0.0
    assert not np.allclose(tick.state.residual.x, planted_x)


def test_home_request_sets_pose_mode() -> None:
    """HOME pose_mode writes a position-loop rate toward home."""
    from dataclasses import replace

    controller = _controller()
    state = replace(
        controller.initial_state(),
        pose_mode=GimbalCommandMode.HOME,
        pose_el_deg=controller.gimbal.home_el_deg,
    )
    tick = controller.outer_step(state, 0.02, _encoder(0.02), None, None, False, False)
    assert tick.state.pose_mode is GimbalCommandMode.HOME
    assert tick.state.r_rad_s > 0.0


def test_science_window_zeros_negative_r_at_min() -> None:
    """TRACKING live does not command r that would leave the science window."""
    controller = _controller()
    state = controller.initial_state()
    iss = _iss()
    centroid = (_BORESIGHT_X, _BORESIGHT_Y + 70.0)
    state, sample = controller.ingest_inference(
        state, _result(1, centroid=centroid), 0.0, 1000.0, iss
    )
    tick = controller.outer_step(state, 0.02, _encoder(0.0), sample, iss, False, False)
    assert tick.state.r_rad_s == 0.0


def test_exit_safe_resets_residual() -> None:
    """EXIT_SAFE cold-starts the residual filter."""
    from dataclasses import replace

    controller = _controller()
    state = controller.initial_state()
    safe = controller.outer_step(state, 0.02, _encoder(0.02), None, None, True, False)
    residual = replace(safe.state.residual, has_measurement=True)
    hot = replace(safe.state, residual=residual)
    cleared = controller.outer_step(hot, 0.04, _encoder(0.04), None, None, False, True)
    assert cleared.state.arbiter.gimbal_state is GimbalState.TRACKING
    assert cleared.state.residual.has_measurement is False
    assert float(cleared.state.residual.x[0]) == 0.0
    assert float(cleared.state.residual.x[1]) == 0.0


def test_rewind_production_zeros_outward_rate_at_sci_min() -> None:
    """Production rate mode (infinite stopping limits) holds r=0 at sci_min in REWIND."""
    controller = _controller()
    iss = _iss()
    state = replace(
        controller.initial_state(),
        arbiter=ArbiterState(
            gimbal_state=GimbalState.REWIND,
            tracked_blobs=(),
            current_target_id=None,
            miss_count=0,
            aggregate_live=False,
            last_observation_s=None,
            loss_handled=True,
            rewind_entered_s=0.0,
        ),
        last_exposure_us=1.0e6,
    )
    tick = controller.outer_step(
        state,
        0.1,
        _encoder(0.1, 0.0),
        None,
        iss,
        False,
        False,
        detailed_plant=False,
    )
    assert tick.state.arbiter.gimbal_state is GimbalState.REWIND
    assert tick.state.r_rad_s == 0.0
    assert math.isfinite(tick.state.r_rad_s)
    assert tick.state.last_omega_t_nom < 0.0
