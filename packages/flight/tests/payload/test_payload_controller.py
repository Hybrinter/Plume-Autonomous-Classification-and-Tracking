"""Tests for the PayloadController cascaded inner/outer cores."""

import math

import numpy as np
from flight.libs.config import ControllerConfig, EphemerisConfig, GimbalConfig, SensorConfig
from flight.libs.messages import BlobMeta, InferenceResultMsg
from flight.libs.types import GimbalCommandMode, GimbalState, MessageType
from flight.payload.control import IssSample, PayloadController, VisionSample

_BORESIGHT_X = 612.0
_BORESIGHT_Y = 512.0


def _controller() -> PayloadController:
    """Build a controller with default controller, sensor, and gimbal geometry."""
    return PayloadController.from_config(
        ControllerConfig(), SensorConfig(), GimbalConfig(), EphemerisConfig()
    )


def _result(frame_id: int, *, centroid: tuple[float, float] | None) -> InferenceResultMsg:
    """Build an InferenceResultMsg, optionally carrying one strong blob at `centroid`."""
    blobs: tuple[BlobMeta, ...] = ()
    if centroid is not None:
        blobs = (
            BlobMeta(
                blob_id=1,
                bbox=(100, 100, 150, 150),
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
    tick = controller.outer_step(state, 0.02, 0.0, None, None, False, False)
    assert tick.state.arbiter.gimbal_state is GimbalState.TRACKING
    assert tick.state.r_rad_s == 0.0
    assert tick.request is None
    assert tick.fault is None


def test_blob_below_boresight_commands_negative_r() -> None:
    """A below-boresight blob snaps e and produces a negative elevation rate."""
    controller = _controller()
    state = controller.initial_state()
    centroid = (_BORESIGHT_X, _BORESIGHT_Y + 70.0)
    state, sample = controller.ingest_inference(state, _result(1, centroid=centroid), 0.0, 1000.0)
    assert sample.z_v is not None
    assert sample.z_v < 0.0
    tick = controller.outer_step(state, 0.02, 0.0, sample, None, False, False)
    assert tick.state.arbiter.gimbal_state is GimbalState.TRACKING
    assert tick.state.residual.has_measurement is True
    assert tick.state.r_rad_s < 0.0
    assert tick.request is None


def test_safe_entry_produces_stow_request() -> None:
    """A commanded SAFE produces a STOW request and latched SAFE state."""
    controller = _controller()
    state = controller.initial_state()
    tick = controller.outer_step(state, 0.02, 0.0, None, None, True, False)
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
    """An IssSample with a stored CoG produces a finite omega_t_nom."""
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
    )
    tick = controller.outer_step(state, 0.02, 0.0, sample, iss, False, False)
    assert math.isfinite(tick.state.last_omega_t_nom)
