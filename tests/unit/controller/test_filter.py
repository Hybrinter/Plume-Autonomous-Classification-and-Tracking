"""Unit tests for pact.controller.filter — ema_update() and EmaFilterState.

Satisfies: §6.2 of PACT_SW_ARCH.md — Controller subsystem unit tests.
REQ-AIML-DATA-007
"""

from __future__ import annotations

# third-party
import pytest

# module under test
from pact.controller.filter import EmaFilterState, ema_update


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_first_frame_no_smoothing() -> None:
    """Uninitialized EMA filter must return the raw centroid directly (no smoothing).

    On the first detection, there is no previous state to blend with, so the output
    must equal the input centroid exactly.
    """
    state = EmaFilterState(centroid=(0.0, 0.0), initialized=False)
    new_centroid = (100.0, 150.0)
    updated = ema_update(state, new_centroid, alpha=0.4)
    assert updated.centroid == pytest.approx(new_centroid), (
        f"Expected raw centroid {new_centroid} on first frame, got {updated.centroid}"
    )
    assert updated.initialized is True


def test_ema_initialized_after_first_update() -> None:
    """After the first update, EmaFilterState.initialized must be True."""
    state = EmaFilterState(centroid=(0.0, 0.0), initialized=False)
    updated = ema_update(state, (10.0, 20.0), alpha=0.4)
    assert updated.initialized is True


def test_ema_formula_applied_when_initialized() -> None:
    """When initialized, EMA formula: smoothed = alpha * new + (1 - alpha) * prev."""
    prev_centroid = (0.0, 0.0)
    new_centroid = (10.0, 20.0)
    alpha = 0.4

    state = EmaFilterState(centroid=prev_centroid, initialized=True)
    updated = ema_update(state, new_centroid, alpha=alpha)

    expected_x = alpha * new_centroid[0] + (1 - alpha) * prev_centroid[0]
    expected_y = alpha * new_centroid[1] + (1 - alpha) * prev_centroid[1]
    assert updated.centroid[0] == pytest.approx(expected_x, rel=1e-5)
    assert updated.centroid[1] == pytest.approx(expected_y, rel=1e-5)


def test_ema_convergence() -> None:
    """EMA must converge to a steady-state input after enough updates.

    After many iterations with the same target centroid, the filtered centroid
    must be within a small epsilon of the target.
    """
    target = (50.0, 75.0)
    alpha = 0.4
    state = EmaFilterState(centroid=(0.0, 0.0), initialized=True)

    # Run 50 iterations with the same target — should converge
    for _ in range(50):
        state = ema_update(state, target, alpha=alpha)

    assert state.centroid[0] == pytest.approx(target[0], abs=0.01), (
        f"EMA did not converge in x: expected {target[0]}, got {state.centroid[0]}"
    )
    assert state.centroid[1] == pytest.approx(target[1], abs=0.01), (
        f"EMA did not converge in y: expected {target[1]}, got {state.centroid[1]}"
    )


@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_ema_alpha_boundary(alpha: float) -> None:
    """EMA update must not raise for any alpha in [0, 1]."""
    state = EmaFilterState(centroid=(10.0, 10.0), initialized=True)
    updated = ema_update(state, (20.0, 30.0), alpha=alpha)
    assert isinstance(updated, EmaFilterState)


def test_ema_alpha_zero_no_movement() -> None:
    """alpha=0 means ignore new input entirely — centroid stays at previous value."""
    prev = (5.0, 5.0)
    state = EmaFilterState(centroid=prev, initialized=True)
    updated = ema_update(state, (100.0, 200.0), alpha=0.0)
    assert updated.centroid == pytest.approx(prev)


def test_ema_alpha_one_instant_update() -> None:
    """alpha=1 means full update to new centroid — previous value ignored."""
    new = (100.0, 200.0)
    state = EmaFilterState(centroid=(5.0, 5.0), initialized=True)
    updated = ema_update(state, new, alpha=1.0)
    assert updated.centroid == pytest.approx(new)


def test_ema_immutable_input_state() -> None:
    """ema_update must not mutate the input EmaFilterState (it is frozen)."""
    state = EmaFilterState(centroid=(10.0, 10.0), initialized=True)
    original_centroid = state.centroid
    ema_update(state, (99.0, 99.0), alpha=0.4)
    # Original state must be unchanged
    assert state.centroid == original_centroid
