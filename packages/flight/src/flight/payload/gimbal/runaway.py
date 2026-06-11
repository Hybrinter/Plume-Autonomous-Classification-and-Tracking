"""Encoder-divergence runaway monitor (pure).

Replaces pixel-inferred runaway with physics: while the controller is commanding
rates (RATE mode), the measured encoder rate between consecutive reads must agree
with the commanded rate within a tolerance. Sustained divergence over strike_limit
consecutive checks raises GIMBAL_RUNAWAY (motor stall, encoder fault, or actuation
without authority). Outside RATE mode -- or when either read is missing or time does
not advance -- the monitor resets rather than guessing (ABSOLUTE/STOW/HOME approach
profiles are driver-internal, so the expected rate is unknown).

Satisfies: REQ-AIML-GIMB-007, REQ-GIMB-HIGH-003.
"""

from __future__ import annotations

# stdlib
import math
from dataclasses import dataclass

# internal
from flight.hal.interfaces import GimbalPosition
from flight.libs.types import FaultCode


@dataclass(frozen=True, slots=True)
class RunawayState:
    """Monitor state threaded across frames.

    Attributes:
        last_pos: The previous encoder read, or None before the first read.
        strike_count: Consecutive divergent checks so far.
    """

    last_pos: GimbalPosition | None
    strike_count: int


INITIAL_RUNAWAY_STATE = RunawayState(last_pos=None, strike_count=0)


def check_runaway(
    state: RunawayState,
    pos: GimbalPosition | None,
    commanded_az_rate_deg_per_s: float,
    commanded_el_rate_deg_per_s: float,
    rate_mode_active: bool,
    tolerance_deg_per_s: float,
    strike_limit: int,
) -> tuple[RunawayState, FaultCode | None]:
    """Compare measured encoder rate against the commanded rate; strike on divergence.

    Inputs:
        state: Previous monitor state (last encoder position and strike counter).
        pos: Current encoder read, or None if the read failed.
        commanded_az_rate_deg_per_s: Azimuth rate most recently sent to the driver (deg/s).
        commanded_el_rate_deg_per_s: Elevation rate most recently sent to the driver (deg/s).
        rate_mode_active: True when the controller is in RATE mode; False resets quietly.
        tolerance_deg_per_s: Maximum allowed divergence between commanded and measured rates.
        strike_limit: Number of consecutive divergent checks before GIMBAL_RUNAWAY is raised.

    Outputs:
        (new_state, fault): Updated monitor state and an optional GIMBAL_RUNAWAY fault code.
        The fault fires when strike_count reaches strike_limit.

    Notes:
        When pos is None, rate_mode_active is False, no previous read exists, or the
        timestamps do not advance, the monitor resets the strike counter to 0 and stores the
        new position. This avoids false positives during ABSOLUTE/STOW/HOME approach profiles
        whose driver-internal velocity is unknown.
    """
    if pos is None:
        return (RunawayState(last_pos=None, strike_count=0), None)
    if (
        not rate_mode_active
        or state.last_pos is None
        or pos.timestamp_s <= state.last_pos.timestamp_s
    ):
        return (RunawayState(last_pos=pos, strike_count=0), None)
    dt = pos.timestamp_s - state.last_pos.timestamp_s
    actual_az = (pos.az_deg - state.last_pos.az_deg) / dt
    actual_el = (pos.el_deg - state.last_pos.el_deg) / dt
    divergence = math.hypot(
        actual_az - commanded_az_rate_deg_per_s, actual_el - commanded_el_rate_deg_per_s
    )
    strikes = state.strike_count + 1 if divergence > tolerance_deg_per_s else 0
    fault = FaultCode.GIMBAL_RUNAWAY if strikes >= strike_limit else None
    return (RunawayState(last_pos=pos, strike_count=strikes), fault)
