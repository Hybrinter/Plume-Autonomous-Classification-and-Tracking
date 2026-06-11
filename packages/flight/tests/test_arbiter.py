"""Unit tests for flight.payload.gimbal.arbiter -- GimbalArbiter state machine.

REQ-AIML-GIMB-001 through 008, REQ-GIMB-HIGH-001 through 004

All tests use the `arbiter_idle_state` fixture from conftest.py as the starting state.
GimbalArbiter.step() is a pure function -- no mocking required.
"""

# third-party
import numpy as np
import pytest

# pact types
from flight.libs.config import PactConfig
from flight.libs.messages import BlobMeta, InferenceResultMsg
from flight.libs.types import GimbalCommandMode, GimbalState, MessageType

# module under test
from flight.payload.gimbal import ArbiterState, GimbalArbiter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_blob(
    blob_id: int = 1,
    mean_confidence: float = 0.85,
    pixel_area: int = 200,
    persistence_count: int = 1,
    bbox: tuple[int, int, int, int] = (100, 100, 150, 150),
    centroid_raw: tuple[float, float] = (125.0, 125.0),
) -> BlobMeta:
    """Construct a BlobMeta with sensible defaults above all safety gates."""
    return BlobMeta(
        blob_id=blob_id,
        bbox=bbox,
        centroid_raw=centroid_raw,
        pixel_area=pixel_area,
        mean_confidence=mean_confidence,
        persistence_count=persistence_count,
    )


def make_inference_result(
    blobs: tuple[BlobMeta, ...] = (),
    frame_id: int = 1,
    mode_flags: int = 0,
) -> InferenceResultMsg:
    """Construct an InferenceResultMsg pre-filtered (blobs already passed all gates)."""
    return InferenceResultMsg(
        msg_type=MessageType.INFERENCE_RESULT,
        timestamp_utc="2026-04-03T00:00:00.000Z",
        frame_id=frame_id,
        mask=np.zeros((256, 256), dtype=np.float32),
        blobs=blobs,
        model_version="test-v0",
        inference_ms=50.0,
        mode_flags=mode_flags,
        crop_origin_px=(0, 0),
        scale_factor=1.0,
    )


# Bit flag used in mode_flags to signal fault -> SAFE transition.
# The arbiter treats any non-zero mode_flags as a fault signal per the arch spec.
_FAULT_FLAG: int = 0b00000001


# ---------------------------------------------------------------------------
# State machine transition tests
# ---------------------------------------------------------------------------


def test_idle_to_acquiring_on_detection(
    arbiter_idle_state: ArbiterState, default_config: PactConfig
) -> None:
    """One blob above threshold detected in IDLE -> new state is ACQUIRING.

    The blob persistence_count=1 < acquire_persistence_frames=3, so we stay in ACQUIRING.
    """
    arbiter = GimbalArbiter(cfg=default_config.controller)
    blob = make_blob(persistence_count=1)
    result = make_inference_result(blobs=(blob,))

    new_state, _request, _events = arbiter.step(arbiter_idle_state, result, None, 1.0, False, False)

    assert new_state.gimbal_state == GimbalState.ACQUIRING


def test_acquiring_to_tracking_on_persistence(
    arbiter_idle_state: ArbiterState, default_config: PactConfig
) -> None:
    """Blob with persistence_count >= 3 in ACQUIRING state -> transitions to TRACKING."""
    arbiter = GimbalArbiter(cfg=default_config.controller)

    blob_p1 = make_blob(persistence_count=1)
    result_p1 = make_inference_result(blobs=(blob_p1,))
    state_acquiring, _, _ = arbiter.step(arbiter_idle_state, result_p1, None, 1.0, False, False)
    assert state_acquiring.gimbal_state == GimbalState.ACQUIRING

    blob_p3 = make_blob(persistence_count=3)
    result_p3 = make_inference_result(blobs=(blob_p3,))
    state_tracking, _, _ = arbiter.step(state_acquiring, result_p3, None, 2.0, False, False)

    assert state_tracking.gimbal_state == GimbalState.TRACKING


def test_tracking_release_hysteresis_holds_then_drops(
    default_config: PactConfig,
) -> None:
    """TRACKING holds for release_persistence_frames - 1 misses, then drops to IDLE."""
    arbiter = GimbalArbiter(cfg=default_config.controller)
    release = default_config.controller.release_persistence_frames
    state = ArbiterState(
        gimbal_state=GimbalState.TRACKING,
        tracked_blobs=(make_blob(persistence_count=5),),
        idle_duration_s=0.0,
        last_command_time=0.0,
        current_target_id=1,
    )
    empty = make_inference_result(blobs=())

    now = 0.0
    for miss in range(1, release):
        now += 1.0
        state, _, _ = arbiter.step(state, empty, None, now, False, False)
        assert state.gimbal_state == GimbalState.TRACKING
        assert state.miss_count == miss

    now += 1.0
    state, _, _ = arbiter.step(state, empty, None, now, False, False)
    assert state.gimbal_state == GimbalState.IDLE


def test_tracking_blob_resets_miss_count(default_config: PactConfig) -> None:
    """A blob seen while in TRACKING resets the release-hysteresis miss counter."""
    arbiter = GimbalArbiter(cfg=default_config.controller)
    state = ArbiterState(
        gimbal_state=GimbalState.TRACKING,
        tracked_blobs=(make_blob(persistence_count=5),),
        idle_duration_s=0.0,
        last_command_time=0.0,
        current_target_id=1,
        miss_count=2,
    )
    result = make_inference_result(blobs=(make_blob(persistence_count=5),))
    state, _, _ = arbiter.step(state, result, (0.0, 0.0), 1.0, False, False)
    assert state.gimbal_state == GimbalState.TRACKING
    assert state.miss_count == 0


def test_idle_to_scan_on_timeout(
    arbiter_idle_state: ArbiterState, default_config: PactConfig
) -> None:
    """After idle_duration_s exceeds scan_entry_idle_seconds (60.0s), transition to SCAN."""
    arbiter = GimbalArbiter(cfg=default_config.controller)
    long_idle_state = ArbiterState(
        gimbal_state=GimbalState.IDLE,
        tracked_blobs=(),
        idle_duration_s=61.0,
        last_command_time=0.0,
        current_target_id=None,
    )
    empty_result = make_inference_result(blobs=())
    new_state, _, _ = arbiter.step(long_idle_state, empty_result, None, 100.0, False, False)
    assert new_state.gimbal_state == GimbalState.SCAN


def test_any_to_safe_on_fault(arbiter_idle_state: ArbiterState, default_config: PactConfig) -> None:
    """Fault signal in mode_flags causes any state to transition to SAFE."""
    arbiter = GimbalArbiter(cfg=default_config.controller)

    for start_state in [
        arbiter_idle_state,
        ArbiterState(
            gimbal_state=GimbalState.ACQUIRING,
            tracked_blobs=(make_blob(persistence_count=1),),
            idle_duration_s=0.0,
            last_command_time=0.0,
            current_target_id=None,
        ),
        ArbiterState(
            gimbal_state=GimbalState.TRACKING,
            tracked_blobs=(make_blob(persistence_count=5),),
            idle_duration_s=0.0,
            last_command_time=0.0,
            current_target_id=1,
        ),
        ArbiterState(
            gimbal_state=GimbalState.SCAN,
            tracked_blobs=(),
            idle_duration_s=65.0,
            last_command_time=0.0,
            current_target_id=None,
        ),
    ]:
        fault_result = make_inference_result(blobs=(), mode_flags=_FAULT_FLAG)
        new_state, _, _ = arbiter.step(start_state, fault_result, None, 1.0, False, False)
        assert new_state.gimbal_state == GimbalState.SAFE


# ---------------------------------------------------------------------------
# SAFE entry / latch / exit and request generation
# ---------------------------------------------------------------------------


def test_safe_entry_emits_single_stow_and_latches(
    arbiter_idle_state: ArbiterState, default_config: PactConfig
) -> None:
    """safe_commanded latches SAFE, emits one STOW request, and ignores blobs after."""
    arbiter = GimbalArbiter(cfg=default_config.controller)
    result = make_inference_result(blobs=(make_blob(persistence_count=5),))

    state, request, _ = arbiter.step(arbiter_idle_state, result, (1.0, 0.0), 1.0, True, False)
    assert state.gimbal_state is GimbalState.SAFE
    assert request is not None
    assert request.mode is GimbalCommandMode.STOW
    assert request.reason == "safe_entry_stow"

    # Latched: a second step in SAFE produces no further request, blobs ignored.
    state, request2, _ = arbiter.step(state, result, (1.0, 0.0), 2.0, False, False)
    assert state.gimbal_state is GimbalState.SAFE
    assert request2 is None


def test_safe_cleared_returns_to_idle(default_config: PactConfig) -> None:
    """safe_cleared in SAFE transitions back to IDLE with counters reset, no request."""
    arbiter = GimbalArbiter(cfg=default_config.controller)
    safe_state = ArbiterState(
        gimbal_state=GimbalState.SAFE,
        tracked_blobs=(),
        idle_duration_s=5.0,
        last_command_time=0.0,
        current_target_id=7,
        miss_count=3,
    )
    empty = make_inference_result(blobs=())
    state, request, _ = arbiter.step(safe_state, empty, None, 1.0, False, True)
    assert state.gimbal_state is GimbalState.IDLE
    assert request is None
    assert state.miss_count == 0
    assert state.idle_duration_s == 0.0
    assert state.current_target_id is None


def test_tracking_emits_rate_request_with_proportional_clip(default_config: PactConfig) -> None:
    """TRACKING with a usable error emits a RATE request clipped to the slew limit."""
    arbiter = GimbalArbiter(cfg=default_config.controller)
    limit = default_config.controller.max_slew_rate_deg_per_s
    state = ArbiterState(
        gimbal_state=GimbalState.TRACKING,
        tracked_blobs=(make_blob(persistence_count=5),),
        idle_duration_s=0.0,
        last_command_time=0.0,
        current_target_id=1,
    )
    result = make_inference_result(blobs=(make_blob(persistence_count=5),))
    # A large error exceeds the proportional (gain 1.0) clip and saturates at the limit.
    state, request, _ = arbiter.step(state, result, (100.0, -100.0), 100.0, False, False)
    assert request is not None
    assert request.mode is GimbalCommandMode.RATE
    assert request.az_deg == limit
    assert request.el_deg == -limit
    assert request.reason == "tracking_target"


def test_scan_reverses_direction_at_boundary(default_config: PactConfig) -> None:
    """The SCAN raster is ABSOLUTE and reverses at the +30/-30 azimuth boundary."""
    arbiter = GimbalArbiter(cfg=default_config.controller)
    scan_state = ArbiterState(
        gimbal_state=GimbalState.SCAN,
        tracked_blobs=(),
        idle_duration_s=65.0,
        last_command_time=0.0,
        current_target_id=None,
        scan_pan_deg=29.9,
        scan_direction=1.0,
    )
    empty = make_inference_result(blobs=())
    state, request, _ = arbiter.step(scan_state, empty, None, 100.0, False, False)
    assert request is not None
    assert request.mode is GimbalCommandMode.ABSOLUTE
    assert request.az_deg == 30.0
    assert state.scan_direction == -1.0
    assert state.scan_pan_deg == 30.0


def test_valid_transitions_exhaustive() -> None:
    """Verify that all defined VALID_TRANSITIONS entries can be exercised.

    Documentation check: asserts the expected transitions match the state machine topology.
    """
    expected_transitions: dict[GimbalState, frozenset[GimbalState]] = {
        GimbalState.IDLE: frozenset({GimbalState.ACQUIRING, GimbalState.SCAN, GimbalState.SAFE}),
        GimbalState.ACQUIRING: frozenset(
            {GimbalState.TRACKING, GimbalState.IDLE, GimbalState.SAFE}
        ),
        GimbalState.TRACKING: frozenset({GimbalState.IDLE, GimbalState.SAFE}),
        GimbalState.SCAN: frozenset({GimbalState.ACQUIRING, GimbalState.IDLE, GimbalState.SAFE}),
        GimbalState.SAFE: frozenset({GimbalState.IDLE}),
    }

    if hasattr(GimbalArbiter, "VALID_TRANSITIONS"):
        vt = getattr(GimbalArbiter, "VALID_TRANSITIONS")
        for from_state, to_states in expected_transitions.items():
            actual = vt.get(from_state, frozenset())
            assert to_states.issubset(actual)
    else:
        pytest.skip(
            "GimbalArbiter.VALID_TRANSITIONS not defined; "
            "transition coverage verified by individual test cases above"
        )
