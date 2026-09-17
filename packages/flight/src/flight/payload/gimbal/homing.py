"""Placeholder assumed-datum homing and STOW rest slew (pure).

INIT creeps from the -45 deg hard stop toward 0 deg (science-window edge) and
back, then declares that contact as -45 deg. There is no vendor index search.
STOW creeps to the configured rest pose.

Satisfies: REQ-OPER-HIGH-002.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, replace

from flight.libs.config import GimbalConfig


class HomingPhase(enum.Enum):
    """INIT inner graph."""

    CREEP_OUT = "CREEP_OUT"
    CREEP_BACK = "CREEP_BACK"
    DATUM = "DATUM"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class HomingState:
    """Immutable INIT homing snapshot."""

    phase: HomingPhase
    phase_started_s: float


@dataclass(frozen=True, slots=True)
class HomingTick:
    """One homing step: new state, commanded rate, and optional failure."""

    state: HomingState
    rate_deg_per_s: float
    failed: bool
    complete: bool


def initial_homing(now: float) -> HomingState:
    """Start CREEP_OUT at `now`."""
    return HomingState(phase=HomingPhase.CREEP_OUT, phase_started_s=now)


def step_homing(state: HomingState, now: float, el_deg: float, cfg: GimbalConfig) -> HomingTick:
    """Advance placeholder homing by one outer tick.

    Args:
        state: Current homing snapshot.
        now: Monotonic seconds.
        el_deg: Measured (or assumed) elevation, degrees.
        cfg: Gimbal envelope and INIT creep config.

    Returns:
        HomingTick with the next state and a rate command.
    """
    lo = cfg.stow_el_deg - 1.0
    hi = cfg.el_science_min_deg + 1.0
    if el_deg < lo or el_deg > hi:
        failed = replace(state, phase=HomingPhase.FAILED)
        return HomingTick(failed, 0.0, failed=True, complete=False)

    rate = cfg.init_creep_rate_deg_per_s
    span_s = cfg.init_creep_span_deg / rate if rate > 0.0 else 0.0

    if state.phase is HomingPhase.COMPLETE:
        return HomingTick(state, 0.0, failed=False, complete=True)
    if state.phase is HomingPhase.FAILED:
        return HomingTick(state, 0.0, failed=True, complete=False)

    if state.phase is HomingPhase.CREEP_OUT:
        elapsed = now - state.phase_started_s
        at_window = el_deg >= cfg.el_science_min_deg - 1e-6
        if elapsed >= span_s or at_window:
            nxt = HomingState(HomingPhase.CREEP_BACK, now)
            return HomingTick(nxt, -rate, failed=False, complete=False)
        return HomingTick(state, rate, failed=False, complete=False)

    if state.phase is HomingPhase.CREEP_BACK:
        elapsed = now - state.phase_started_s
        at_stop = el_deg <= cfg.stow_el_deg + cfg.stow_arrive_tol_deg
        if at_stop or elapsed >= span_s + cfg.init_press_timeout_s:
            nxt = HomingState(HomingPhase.DATUM, now)
            return HomingTick(nxt, 0.0, failed=False, complete=False)
        return HomingTick(state, -rate, failed=False, complete=False)

    # DATUM
    done = HomingState(HomingPhase.COMPLETE, now)
    return HomingTick(done, 0.0, failed=False, complete=True)


def stow_rate_deg_per_s(el_deg: float, cfg: GimbalConfig) -> tuple[float, bool]:
    """Rate toward rest and whether arrival is confirmed.

    Args:
        el_deg: Current elevation, degrees.
        cfg: Stow pose, creep rate, and arrival tolerance.

    Returns:
        (rate_deg_per_s, arrived).
    """
    err = cfg.stow_el_deg - el_deg
    if abs(err) <= cfg.stow_arrive_tol_deg:
        return 0.0, True
    rate = cfg.init_creep_rate_deg_per_s
    return (-rate if err < 0.0 else rate), False
