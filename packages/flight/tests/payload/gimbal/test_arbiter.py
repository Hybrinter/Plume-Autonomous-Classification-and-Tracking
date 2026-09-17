"""Unit tests for flight.payload.gimbal.arbiter -- TRACKING / REWIND / SAFE.

REQ-AIML-GIMB-001 through 008, REQ-GIMB-HIGH-001 through 004
"""

from flight.libs.config import PactConfig
from flight.libs.messages import BlobMeta
from flight.libs.types import GimbalCommandMode, GimbalState
from flight.payload.gimbal import ArbiterState, GimbalArbiter

_FAULT_FLAG: int = 0b00000001


def _arbiter(config: PactConfig) -> GimbalArbiter:
    """Build an arbiter from controller + gimbal slices of PactConfig."""
    return GimbalArbiter(config.controller.arbiter, config.gimbal)


def make_blob(
    blob_id: int = 1,
    mean_confidence: float = 0.85,
    pixel_area: int = 200,
) -> BlobMeta:
    """Construct a BlobMeta above all safety gates."""
    return BlobMeta(
        blob_id=blob_id,
        bbox=(100, 100, 150, 150),
        centroid_raw=(125.0, 125.0),
        pixel_area=pixel_area,
        mean_confidence=mean_confidence,
        persistence_count=1,
    )


def test_plume_enters_tracking(
    arbiter_tracking_state: ArbiterState, default_config: PactConfig
) -> None:
    """An accepted aggregate enters TRACKING immediately without selecting a blob."""
    arbiter = _arbiter(default_config)
    new_state, request, events = arbiter.step(
        arbiter_tracking_state,
        (make_blob(),),
        now=1.0,
        safe_commanded=False,
        safe_cleared=False,
        el_deg=10.0,
    )
    assert new_state.gimbal_state is GimbalState.TRACKING
    assert new_state.aggregate_live is True
    assert new_state.current_target_id is None
    assert new_state.last_observation_s == 1.0
    assert new_state.miss_count == 0
    assert request is None
    assert events == []


def test_misses_below_limb_enter_rewind(
    arbiter_tracking_state: ArbiterState, default_config: PactConfig
) -> None:
    """release_persistence_frames empty vision samples below the limb enter REWIND."""
    arbiter = _arbiter(default_config)
    state = arbiter_tracking_state
    persist = default_config.controller.arbiter.release_persistence_frames
    request = None
    for i in range(persist):
        state, request, events = arbiter.step(
            state,
            (),
            now=float(i + 1),
            safe_commanded=False,
            safe_cleared=False,
            el_deg=10.0,
        )
    assert state.gimbal_state is GimbalState.REWIND
    assert state.rewind_entered_s == float(persist)
    assert request is None
    assert events[-1].payload["to"] == GimbalState.REWIND.value


def test_misses_at_limb_stay_tracking(
    arbiter_tracking_state: ArbiterState, default_config: PactConfig
) -> None:
    """Loss at the science limb stays TRACKING (wait with r = 0)."""
    arbiter = _arbiter(default_config)
    state = arbiter_tracking_state
    persist = default_config.controller.arbiter.release_persistence_frames
    limb = default_config.gimbal.el_science_max_deg
    for i in range(persist):
        state, _request, _events = arbiter.step(
            state,
            (),
            now=float(i + 1),
            safe_commanded=False,
            safe_cleared=False,
            el_deg=limb,
        )
    assert state.gimbal_state is GimbalState.TRACKING
    assert state.miss_count == 0
    assert state.current_target_id is None


def test_coast_does_not_increment_miss(
    arbiter_tracking_state: ArbiterState, default_config: PactConfig
) -> None:
    """Outer ticks without a vision sample leave miss_count unchanged."""
    arbiter = _arbiter(default_config)
    seeded, _request, _events = arbiter.step(
        arbiter_tracking_state,
        (make_blob(),),
        now=1.0,
        safe_commanded=False,
        safe_cleared=False,
        el_deg=10.0,
    )
    coast, _request, _events = arbiter.step(
        seeded,
        (),
        now=1.02,
        safe_commanded=False,
        safe_cleared=False,
        el_deg=10.0,
        vision_updated=False,
    )
    assert coast.gimbal_state is GimbalState.TRACKING
    assert coast.miss_count == 0
    assert coast.aggregate_live is True
    assert coast.current_target_id is None


def test_silent_vision_stall_expires_coast(
    arbiter_tracking_state: ArbiterState, default_config: PactConfig
) -> None:
    """No arriving empty packets cannot preserve a live aggregate indefinitely."""
    arbiter = _arbiter(default_config)
    seeded, _request, _events = arbiter.step(
        arbiter_tracking_state,
        (make_blob(),),
        now=1.0,
        safe_commanded=False,
        safe_cleared=False,
        el_deg=10.0,
    )
    expired, _request, events = arbiter.step(
        seeded,
        (),
        now=1.0 + default_config.controller.arbiter.max_observation_age_s + 0.001,
        safe_commanded=False,
        safe_cleared=False,
        el_deg=10.0,
        vision_updated=False,
    )
    assert expired.gimbal_state is GimbalState.REWIND
    assert expired.aggregate_live is False
    assert events[-1].payload["to"] == GimbalState.REWIND.value


def test_uncertainty_gate_can_revoke_prediction_only_coast(
    arbiter_tracking_state: ArbiterState, default_config: PactConfig
) -> None:
    """The estimator can later terminate a coast before its age ceiling."""
    arbiter = _arbiter(default_config)
    seeded, _request, _events = arbiter.step(
        arbiter_tracking_state,
        (make_blob(),),
        now=1.0,
        safe_commanded=False,
        safe_cleared=False,
        el_deg=10.0,
    )
    revoked, _request, _events = arbiter.step(
        seeded,
        (),
        now=1.01,
        safe_commanded=False,
        safe_cleared=False,
        el_deg=10.0,
        vision_updated=False,
        coast_permitted=False,
    )
    assert revoked.gimbal_state is GimbalState.REWIND
    assert revoked.aggregate_live is False


def test_rewind_plume_returns_to_tracking(default_config: PactConfig) -> None:
    """A plume during REWIND returns to TRACKING immediately."""
    arbiter = _arbiter(default_config)
    rewind = ArbiterState(
        gimbal_state=GimbalState.REWIND,
        tracked_blobs=(),
        current_target_id=None,
        miss_count=0,
    )
    new_state, request, events = arbiter.step(
        rewind,
        (make_blob(blob_id=7),),
        now=2.0,
        safe_commanded=False,
        safe_cleared=False,
        el_deg=10.0,
    )
    assert new_state.gimbal_state is GimbalState.TRACKING
    assert new_state.rewind_entered_s is None
    assert new_state.aggregate_live is True
    assert new_state.current_target_id is None
    assert request is None
    assert events[-1].payload["to"] == GimbalState.TRACKING.value


def test_rewind_at_limb_returns_to_tracking(default_config: PactConfig) -> None:
    """Arrival at the science limb with no plume returns REWIND to TRACKING."""
    arbiter = _arbiter(default_config)
    rewind = ArbiterState(
        gimbal_state=GimbalState.REWIND,
        tracked_blobs=(),
        current_target_id=None,
        miss_count=0,
    )
    limb = default_config.gimbal.el_science_max_deg
    new_state, request, _events = arbiter.step(
        rewind,
        (),
        now=2.0,
        safe_commanded=False,
        safe_cleared=False,
        el_deg=limb,
    )
    assert new_state.gimbal_state is GimbalState.TRACKING
    assert request is None


def test_safe_latches_and_stows(
    arbiter_tracking_state: ArbiterState, default_config: PactConfig
) -> None:
    """SAFE entry issues STOW and stays latched until cleared."""
    arbiter = _arbiter(default_config)
    new_state, request, events = arbiter.step(
        arbiter_tracking_state,
        (make_blob(),),
        now=1.0,
        safe_commanded=True,
        safe_cleared=False,
        el_deg=10.0,
    )
    assert new_state.gimbal_state is GimbalState.SAFE
    assert request is not None
    assert request.mode is GimbalCommandMode.STOW
    assert request.el_deg == default_config.gimbal.stow_el_deg
    assert events[-1].payload["to"] == GimbalState.SAFE.value

    held, held_request, _events = arbiter.step(
        new_state,
        (make_blob(),),
        now=2.0,
        safe_commanded=False,
        safe_cleared=False,
        el_deg=10.0,
    )
    assert held.gimbal_state is GimbalState.SAFE
    assert held_request is None


def test_mode_flags_latch_safe(
    arbiter_tracking_state: ArbiterState, default_config: PactConfig
) -> None:
    """Nonzero inference mode_flags latches SAFE."""
    arbiter = _arbiter(default_config)
    new_state, request, _events = arbiter.step(
        arbiter_tracking_state,
        (make_blob(),),
        now=1.0,
        safe_commanded=False,
        safe_cleared=False,
        el_deg=10.0,
        mode_flags=_FAULT_FLAG,
    )
    assert new_state.gimbal_state is GimbalState.SAFE
    assert request is not None
    assert request.mode is GimbalCommandMode.STOW


def test_safe_clear_returns_to_tracking(default_config: PactConfig) -> None:
    """A non-SAFE mode change exits SAFE to TRACKING."""
    arbiter = _arbiter(default_config)
    safe = ArbiterState(
        gimbal_state=GimbalState.SAFE,
        tracked_blobs=(),
        current_target_id=None,
        miss_count=0,
    )
    new_state, request, events = arbiter.step(
        safe,
        (),
        now=3.0,
        safe_commanded=False,
        safe_cleared=True,
        el_deg=-45.0,
    )
    assert new_state.gimbal_state is GimbalState.TRACKING
    assert request is None
    assert events[-1].payload["from"] == GimbalState.SAFE.value


def test_transition_uses_injected_timestamp(
    arbiter_tracking_state: ArbiterState, default_config: PactConfig
) -> None:
    """Arbiter transition telemetry uses the injected timestamp, not wall clock."""
    arbiter = _arbiter(default_config)
    _state, _request, events = arbiter.step(
        arbiter_tracking_state,
        (make_blob(),),
        now=1.0,
        safe_commanded=True,
        safe_cleared=False,
        el_deg=10.0,
        timestamp_utc="2026-09-08T00:00:00.000Z",
    )
    assert events[-1].timestamp_utc == "2026-09-08T00:00:00.000Z"


def test_legal_transitions(default_config: PactConfig) -> None:
    """Allowed mode edges are TRACKING↔REWIND and either↔SAFE."""
    allowed = {
        GimbalState.TRACKING: frozenset(
            {GimbalState.TRACKING, GimbalState.REWIND, GimbalState.SAFE}
        ),
        GimbalState.REWIND: frozenset({GimbalState.REWIND, GimbalState.TRACKING, GimbalState.SAFE}),
        GimbalState.SAFE: frozenset({GimbalState.SAFE, GimbalState.TRACKING}),
    }
    assert set(GimbalState) == set(allowed)
    assert GimbalCommandMode.ABSOLUTE in GimbalCommandMode
    del default_config
