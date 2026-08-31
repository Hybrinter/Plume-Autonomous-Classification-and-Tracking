"""Tests for the PayloadController ingest / outer / inner composition."""

import math

import numpy as np
from flight.hal.interfaces import GimbalPosition
from flight.libs.config import ControllerConfig, SensorConfig
from flight.libs.messages import BlobMeta, InferenceResultMsg
from flight.libs.types import GimbalCommandMode, GimbalState, MessageType
from flight.payload.control import ControlState, PayloadController
from flight.payload.gimbal import GimbalRequest

# Default geometry: 1024 sensor -> 512 plane, boresight at (256, 256).
_BORESIGHT = 256.0


def _controller() -> PayloadController:
    """Build a controller with the default controller + sensor geometry."""
    return PayloadController.from_config(ControllerConfig(), SensorConfig())


def _result(
    frame_id: int,
    *,
    centroid: tuple[float, float] | None,
    area: int = 200,
) -> InferenceResultMsg:
    """Build an InferenceResultMsg, optionally carrying one strong blob at `centroid`."""
    mask = np.zeros((16, 16), dtype=np.float32)
    blobs: tuple[BlobMeta, ...] = ()
    if centroid is not None:
        blobs = (
            BlobMeta(
                blob_id=1,
                bbox=(100, 100, 150, 150),
                centroid_raw=centroid,
                pixel_area=area,
                mean_confidence=0.85,
                persistence_count=1,
            ),
        )
    return InferenceResultMsg(
        msg_type=MessageType.INFERENCE_RESULT,
        timestamp_utc="2026-06-01T00:00:00.000Z",
        frame_id=frame_id,
        mask=mask,
        blobs=blobs,
        model_version="test",
        inference_ms=0.0,
        mode_flags=0,
        crop_origin_px=(0, 0),
        scale_factor=1.0,
    )


def _pos(az: float = 0.0, el: float = 0.0, t: float = 0.0) -> GimbalPosition:
    """Build a zero-noise encoder sample."""
    return GimbalPosition(az_deg=az, el_deg=el, timestamp_s=t)


def _track_to_tracking(
    controller: PayloadController,
    state: ControlState,
    centroid: tuple[float, float],
    now: float,
) -> tuple[ControlState, float]:
    """Ingest enough persistent frames to enter TRACKING."""
    acquire = controller.cfg.acquire_persistence_frames
    for frame_id in range(1, acquire + 1):
        now += 0.1
        state = controller.ingest_vision(state, _result(frame_id, centroid=centroid), now)
        state, _request, _events, _fault = controller.step_outer(
            state, _pos(t=now), now, 0.1, False, False
        )
    return state, now


def test_initial_state_is_idle() -> None:
    """The controller starts IDLE with no tracked blobs and r = 0."""
    state = _controller().initial_state()
    assert state.arbiter.gimbal_state is GimbalState.IDLE
    assert state.arbiter.tracked_blobs == ()
    assert state.seen_vision is False
    assert state.r_az_deg_s == 0.0
    assert state.r_el_deg_s == 0.0


def test_no_detection_stays_idle_no_command() -> None:
    """With no blobs, the controller stays IDLE and issues no request."""
    controller = _controller()
    state = controller.initial_state()
    state, request, _events, fault = controller.step(
        state, _result(1, centroid=None), 1.0, None, False, False
    )
    assert state.arbiter.gimbal_state is GimbalState.IDLE
    assert request is None
    assert fault is None


def test_r_is_zero_before_first_vision() -> None:
    """Outer ticks with encoder only hold r = 0 until the first vision sample."""
    controller = _controller()
    state = controller.initial_state()
    now = 0.0
    for i in range(5):
        now += 0.1
        state, _request, _events, _fault = controller.step_outer(
            state, _pos(t=now), now, 0.1, False, False
        )
        assert state.r_az_deg_s == 0.0
        assert state.r_el_deg_s == 0.0
        assert state.seen_vision is False


def test_persistent_blob_progresses_to_tracking_and_commands() -> None:
    """A stable off-center blob over frames drives TRACKING and issues a RATE request."""
    controller = _controller()
    state = controller.initial_state()
    centroid = (_BORESIGHT + 70.0, _BORESIGHT + 70.0)
    now = 0.0
    last_rate: GimbalRequest | None = None
    for frame_id in range(1, 9):
        now += 1.0
        state, request, _events, _fault = controller.step(
            state, _result(frame_id, centroid=centroid), now, _pos(t=now), False, False
        )
        if request is not None and request.mode is GimbalCommandMode.RATE:
            last_rate = request
    assert state.arbiter.gimbal_state is GimbalState.TRACKING
    assert last_rate is not None
    assert last_rate.az_deg > 0.0
    assert last_rate.el_deg < 0.0
    assert state.r_az_deg_s > 0.0
    assert state.r_el_deg_s < 0.0


def test_outer_ticks_without_new_frame_still_slew() -> None:
    """After ingest, further outer ticks with no new frame keep a nonzero r."""
    controller = _controller()
    state = controller.initial_state()
    centroid = (_BORESIGHT + 70.0, _BORESIGHT + 70.0)
    state, now = _track_to_tracking(controller, state, centroid, 0.0)
    assert state.arbiter.gimbal_state is GimbalState.TRACKING
    assert state.seen_vision is True
    assert state.r_az_deg_s > 0.0
    for _ in range(5):
        now += 0.05
        state, request, _events, _fault = controller.step_outer(
            state, _pos(t=now), now, 0.05, False, False
        )
        assert state.arbiter.gimbal_state is GimbalState.TRACKING
        assert state.pending_vision_frame is False
        assert state.r_az_deg_s > 0.0
        assert request is not None
        assert request.mode is GimbalCommandMode.RATE


def test_safe_entry_produces_stow_request_and_latched_state() -> None:
    """A commanded SAFE produces a STOW request and a latched SAFE arbiter state."""
    controller = _controller()
    state = controller.initial_state()
    state, request, _events, _fault = controller.step(
        state, _result(1, centroid=None), 1.0, None, True, False
    )
    assert request is not None
    assert request.mode is GimbalCommandMode.STOW
    assert state.arbiter.gimbal_state is GimbalState.SAFE
    assert state.servo.integral_az == 0.0
    assert state.servo.integral_el == 0.0


def test_returns_bundled_control_state() -> None:
    """step() threads a ControlState bundling arbiter, EMA, Kalman, servo, and ring."""
    controller = _controller()
    state = controller.initial_state()
    centroid = (_BORESIGHT + 70.0, _BORESIGHT + 70.0)
    state, _request, _events, _fault = controller.step(
        state, _result(1, centroid=centroid), 1.0, None, False, False
    )
    assert isinstance(state, ControlState)
    assert state.ema.initialized is True
    assert state.pending_vision_frame is False
    assert state.seen_vision is True


def test_ingest_uses_area_weighted_com() -> None:
    """Two gated blobs contribute to the mailbox in proportion to pixel area."""
    controller = _controller()
    state = controller.initial_state()
    left = BlobMeta(
        blob_id=1,
        bbox=(10, 10, 20, 20),
        centroid_raw=(_BORESIGHT - 40.0, _BORESIGHT),
        pixel_area=100,
        mean_confidence=0.9,
        persistence_count=1,
    )
    right = BlobMeta(
        blob_id=2,
        bbox=(200, 10, 220, 20),
        centroid_raw=(_BORESIGHT + 80.0, _BORESIGHT),
        pixel_area=200,
        mean_confidence=0.9,
        persistence_count=1,
    )
    result = InferenceResultMsg(
        msg_type=MessageType.INFERENCE_RESULT,
        timestamp_utc="2026-06-01T00:00:00.000Z",
        frame_id=1,
        mask=np.zeros((16, 16), dtype=np.float32),
        blobs=(left, right),
        model_version="test",
        inference_ms=0.0,
        mode_flags=0,
        crop_origin_px=(0, 0),
        scale_factor=1.0,
    )
    state = controller.ingest_vision(state, result, 1.0)
    # CoM x = (100*(-40) + 200*(80)) / 300 = 40 px right of boresight -> +0.8 deg.
    assert state.vis_z_az is not None
    assert abs(state.vis_z_az - 0.8) < 1e-9
    assert state.vis_z_el is not None
    assert abs(state.vis_z_el) < 1e-9


def test_step_inner_converts_degrees_and_returns_torque() -> None:
    """step_inner converts r from deg/s to rad/s and returns a torque pair."""
    controller = _controller()
    state = controller.initial_state()
    state = ControlState(
        arbiter=state.arbiter,
        ema=state.ema,
        kalman=state.kalman,
        servo=state.servo,
        ring=state.ring,
        r_az_deg_s=2.0,
        r_el_deg_s=0.0,
        seen_vision=True,
        vis_z_az=None,
        vis_z_el=None,
        vis_t_shutter=None,
        last_outer_time_s=0.0,
        vision_blobs=(),
        vision_mode_flags=0,
        pending_vision_frame=False,
    )
    new_state, tau = controller.step_inner(state, _pos(az=0.0, el=0.0, t=0.01), 0.01)
    assert math.isfinite(tau[0])
    assert math.isfinite(tau[1])
    assert new_state.servo.last_theta_az_rad is not None
