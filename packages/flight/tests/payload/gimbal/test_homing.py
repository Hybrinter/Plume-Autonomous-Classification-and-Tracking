"""Unit tests for placeholder INIT homing and STOW slew."""

from flight.libs.config import GimbalConfig
from flight.payload.gimbal.homing import HomingPhase, initial_homing, step_homing, stow_rate_deg_per_s


def test_homing_creep_out_then_back_then_complete() -> None:
    """Assumed-datum homing walks CREEP_OUT, CREEP_BACK, DATUM, COMPLETE."""
    cfg = GimbalConfig()
    state = initial_homing(0.0)
    tick = step_homing(state, 0.0, -45.0, cfg)
    assert tick.state.phase is HomingPhase.CREEP_OUT
    assert tick.rate_deg_per_s > 0.0

    span_s = cfg.init_creep_span_deg / cfg.init_creep_rate_deg_per_s
    tick = step_homing(tick.state, span_s, 0.0, cfg)
    assert tick.state.phase is HomingPhase.CREEP_BACK
    assert tick.rate_deg_per_s < 0.0

    tick = step_homing(tick.state, span_s + 0.1, -45.0, cfg)
    assert tick.state.phase is HomingPhase.DATUM

    tick = step_homing(tick.state, span_s + 0.2, -45.0, cfg)
    assert tick.complete is True
    assert tick.state.phase is HomingPhase.COMPLETE


def test_homing_fails_outside_envelope() -> None:
    """Elevation past the science-window edge fails INIT."""
    cfg = GimbalConfig()
    state = initial_homing(0.0)
    tick = step_homing(state, 0.0, 5.0, cfg)
    assert tick.failed is True
    assert tick.state.phase is HomingPhase.FAILED


def test_stow_rate_arrives_at_rest() -> None:
    """STOW slew reports arrival inside the rest tolerance."""
    cfg = GimbalConfig()
    rate, arrived = stow_rate_deg_per_s(-45.0, cfg)
    assert arrived is True
    assert rate == 0.0
    rate, arrived = stow_rate_deg_per_s(0.0, cfg)
    assert arrived is False
    assert rate < 0.0
