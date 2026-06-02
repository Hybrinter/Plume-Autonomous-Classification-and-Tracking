"""Tests for the PayloadController pure control composition."""

import numpy as np
from flight.libs.config import ControllerConfig
from flight.libs.messages import BlobMeta, InferenceResultMsg
from flight.libs.types import GimbalState, MessageType
from flight.payload.control import ControlState, PayloadController


def _result_with_blob(frame_id: int, *, with_blob: bool) -> InferenceResultMsg:
    """Build an InferenceResultMsg, optionally carrying one strong, stable blob."""
    mask = np.zeros((16, 16), dtype=np.float32)
    blobs: tuple[BlobMeta, ...] = ()
    if with_blob:
        blobs = (
            BlobMeta(
                blob_id=1,
                bbox=(100, 100, 150, 150),
                centroid_raw=(125.0, 125.0),
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
    )


def test_initial_state_is_idle() -> None:
    """The controller starts IDLE with no tracked blobs."""
    controller = PayloadController.from_config(ControllerConfig())
    state = controller.initial_state()
    assert state.arbiter.gimbal_state is GimbalState.IDLE
    assert state.arbiter.tracked_blobs == ()


def test_no_detection_stays_idle_no_command() -> None:
    """With no blobs, the controller stays IDLE and issues no command."""
    controller = PayloadController.from_config(ControllerConfig())
    state = controller.initial_state()
    state, command, _events = controller.step(state, _result_with_blob(1, with_blob=False), now=1.0)
    assert state.arbiter.gimbal_state is GimbalState.IDLE
    assert command is None


def test_persistent_blob_progresses_to_tracking_and_commands() -> None:
    """A stable, strong blob over several frames drives the FSM to TRACKING and issues a command."""
    controller = PayloadController.from_config(ControllerConfig())
    state = controller.initial_state()
    now = 0.0
    saw_command = False
    for frame_id in range(1, 9):
        now += 1.0
        state, command, _events = controller.step(
            state, _result_with_blob(frame_id, with_blob=True), now
        )
        if command is not None:
            saw_command = True
    assert state.arbiter.gimbal_state is GimbalState.TRACKING
    assert saw_command


def test_returns_bundled_control_state() -> None:
    """step() threads a ControlState that bundles arbiter, EMA, and Kalman sub-states."""
    controller = PayloadController.from_config(ControllerConfig())
    state = controller.initial_state()
    state, _command, _events = controller.step(state, _result_with_blob(1, with_blob=True), now=1.0)
    assert isinstance(state, ControlState)
    assert state.ema.initialized is True
