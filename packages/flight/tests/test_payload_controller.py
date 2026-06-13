"""Tests for the PayloadController pure control composition (boresight-error space)."""

import numpy as np
from flight.hal.interfaces import GimbalPosition
from flight.libs.config import ControllerConfig, SensorConfig
from flight.libs.messages import BlobMeta, InferenceResultMsg
from flight.libs.types import FaultCode, GimbalCommandMode, GimbalState, MessageType
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
                pixel_area=200,
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


def test_initial_state_is_idle() -> None:
    """The controller starts IDLE with no tracked blobs and a fresh runaway monitor."""
    state = _controller().initial_state()
    assert state.arbiter.gimbal_state is GimbalState.IDLE
    assert state.arbiter.tracked_blobs == ()
    assert state.deadband_strikes == 0
    assert state.runaway.last_pos is None


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


def test_persistent_blob_progresses_to_tracking_and_commands() -> None:
    """A stable off-center blob over frames drives TRACKING and issues a RATE request."""
    controller = _controller()
    state = controller.initial_state()
    # 70 px off boresight -> ~99 px displacement: above min_deadband, below max_deadband.
    centroid = (_BORESIGHT + 70.0, _BORESIGHT + 70.0)
    now = 0.0
    last_rate: GimbalRequest | None = None
    for frame_id in range(1, 9):
        now += 1.0
        state, request, _events, _fault = controller.step(
            state, _result(frame_id, centroid=centroid), now, None, False, False
        )
        if request is not None and request.mode is GimbalCommandMode.RATE:
            last_rate = request
    assert state.arbiter.gimbal_state is GimbalState.TRACKING
    assert last_rate is not None
    # Target right (+x) and below (+y) of boresight -> slew toward it: +az and -el.
    assert last_rate.az_deg > 0.0
    assert last_rate.el_deg < 0.0


def test_deadband_below_min_suppresses_rate_command() -> None:
    """A target inside the minimum deadband (near boresight) never issues a RATE command."""
    controller = _controller()
    state = controller.initial_state()
    centroid = (_BORESIGHT + 2.0, _BORESIGHT + 2.0)  # ~2.8 px < min_deadband_px=20
    now = 0.0
    for frame_id in range(1, 9):
        now += 1.0
        state, request, _events, _fault = controller.step(
            state, _result(frame_id, centroid=centroid), now, None, False, False
        )
        assert not (request is not None and request.mode is GimbalCommandMode.RATE)


def test_max_deadband_strikes_raise_runaway_fault() -> None:
    """Displacement above max_deadband_px for the strike count raises GIMBAL_RUNAWAY."""
    controller = _controller()
    state = controller.initial_state()
    # 200 px off each axis -> ~283 px displacement > max_deadband_px=250.
    centroid = (_BORESIGHT + 200.0, _BORESIGHT + 200.0)
    strike_limit = ControllerConfig().max_deadband_strike_count
    now = 0.0
    fault: FaultCode | None = None
    for frame_id in range(1, strike_limit + 1):
        now += 1.0
        state, _request, _events, fault = controller.step(
            state, _result(frame_id, centroid=centroid), now, None, False, False
        )
    assert fault is FaultCode.GIMBAL_RUNAWAY


def test_runaway_fault_from_stalled_encoder_while_commanding() -> None:
    """A stalled encoder while a RATE was commanded last frame raises GIMBAL_RUNAWAY."""
    cfg = ControllerConfig()
    controller = PayloadController.from_config(cfg, SensorConfig())
    # Seed a state that commanded a 2 deg/s azimuth rate last frame.
    base = controller.initial_state()
    state = ControlState(
        arbiter=base.arbiter,
        ema=base.ema,
        kalman=base.kalman,
        runaway=base.runaway,
        deadband_strikes=0,
        commanded_az_rate_deg_per_s=2.0,
        commanded_el_rate_deg_per_s=0.0,
    )
    fault: FaultCode | None = None
    # Encoder reports no motion across consecutive timestamps -> divergence strikes.
    for i in range(0, cfg.runaway_strike_count + 1):
        pos = GimbalPosition(az_deg=0.0, el_deg=0.0, timestamp_s=float(i))
        state, _request, _events, fault = controller.step(
            state, _result(i + 1, centroid=None), float(i), pos, False, False
        )
        # Keep the commanded rate non-zero so the monitor stays armed each frame.
        state = ControlState(
            arbiter=state.arbiter,
            ema=state.ema,
            kalman=state.kalman,
            runaway=state.runaway,
            deadband_strikes=state.deadband_strikes,
            commanded_az_rate_deg_per_s=2.0,
            commanded_el_rate_deg_per_s=0.0,
        )
    assert fault is FaultCode.GIMBAL_RUNAWAY


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


def test_returns_bundled_control_state() -> None:
    """step() threads a ControlState bundling arbiter, EMA, Kalman, and runaway sub-states."""
    controller = _controller()
    state = controller.initial_state()
    centroid = (_BORESIGHT + 70.0, _BORESIGHT + 70.0)
    state, _request, _events, _fault = controller.step(
        state, _result(1, centroid=centroid), 1.0, None, False, False
    )
    assert isinstance(state, ControlState)
    assert state.ema.initialized is True
