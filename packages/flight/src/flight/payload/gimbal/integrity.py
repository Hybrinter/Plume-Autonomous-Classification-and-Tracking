"""Light pointing-integrity detector (pure). NaN, encoder freeze, lock-fight.

Not the deleted RATE-mode runaway monitor. Strikes accumulate across inner ticks.
NaN torque or rate is immediate. Encoder freeze: commanded |r| above r_min while
the measured encoder rate is a small fraction of |r|. Lock-fight: lock engaged
and hold-pose motion above the pin threshold (axis walking against the pin).
Hold-pose motion is encoder displacement since lock engage over elapsed time,
not the 1 kHz polynomial y_m (encoder noise at T_in exceeds the pin rate).

Satisfies: REQ-GIMB-HIGH-003.
"""

from __future__ import annotations

# stdlib
import math
from dataclasses import dataclass

# internal
from flight.libs.config import IntegrityConfig
from flight.libs.types import FaultCode


@dataclass(frozen=True, slots=True)
class IntegrityResult:
    """One integrity check.

    Attributes:
        freeze_strikes: Updated encoder-freeze strike count.
        lock_fight_strikes: Updated lock-fight strike count.
        fault: GIMBAL_RUNAWAY when a trip fires, else None.
    """

    freeze_strikes: int
    lock_fight_strikes: int
    fault: FaultCode | None


def lock_hold_rate(
    locked: bool,
    theta_rad: float,
    now: float,
    ref_theta_rad: float | None,
    ref_s: float | None,
) -> tuple[float, float | None, float | None]:
    """Hold-pose motion since lock engage, and the updated pose reference.

    Inputs:
        locked: True when the launch lock inhibits motion.
        theta_rad: Current encoder elevation, rad.
        now: Monotonic seconds of this inner tick.
        ref_theta_rad: Encoder elevation latched at lock engage, or None.
        ref_s: Monotonic seconds of that latch, or None.

    Outputs:
        tuple[float, float | None, float | None]: Motion rad/s, new ref angle,
        new ref time. Motion is 0 and the ref is cleared when the lock is open.
        Motion is 0 on the latching tick.
    """
    if not locked:
        return 0.0, None, None
    if ref_theta_rad is None or ref_s is None:
        return 0.0, theta_rad, now
    dt_s = now - ref_s
    if dt_s <= 1e-9:
        return 0.0, ref_theta_rad, ref_s
    return (theta_rad - ref_theta_rad) / dt_s, ref_theta_rad, ref_s


def check_integrity(
    cfg: IntegrityConfig,
    r_rad_s: float,
    y_m: float,
    tau_nm: float,
    encoder_rate_rad_s: float,
    lock_engaged: bool,
    freeze_strikes: int,
    lock_fight_strikes: int,
    lock_motion_rad_s: float = 0.0,
) -> IntegrityResult:
    """Advance strike counters and trip GIMBAL_RUNAWAY when a rule fires.

    Inputs:
        cfg: Integrity thresholds and strike limits.
        r_rad_s: Rate reference, rad/s.
        y_m: Encoder-rate estimate, rad/s.
        tau_nm: Torque command, N·m.
        encoder_rate_rad_s: Raw two-sample encoder rate, rad/s.
        lock_engaged: True when the launch lock inhibits motion.
        freeze_strikes, lock_fight_strikes: Prior counts.
        lock_motion_rad_s: Hold-pose rate since lock engage, rad/s.

    Outputs:
        IntegrityResult: Updated counts and optional fault.
    """
    if not (
        math.isfinite(r_rad_s)
        and math.isfinite(y_m)
        and math.isfinite(tau_nm)
        and math.isfinite(lock_motion_rad_s)
    ):
        return IntegrityResult(
            freeze_strikes=freeze_strikes,
            lock_fight_strikes=lock_fight_strikes,
            fault=FaultCode.GIMBAL_RUNAWAY,
        )

    freeze = freeze_strikes
    if abs(r_rad_s) > cfg.r_min_rad_s and abs(encoder_rate_rad_s) < cfg.encoder_rate_ratio * abs(
        r_rad_s
    ):
        freeze += 1
    else:
        freeze = 0

    fight = lock_fight_strikes
    if lock_engaged and abs(lock_motion_rad_s) > cfg.lock_fight_rad_s:
        fight += 1
    else:
        fight = 0

    fault: FaultCode | None = None
    if freeze >= cfg.freeze_strikes or fight >= cfg.lock_fight_strikes:
        fault = FaultCode.GIMBAL_RUNAWAY
    return IntegrityResult(freeze_strikes=freeze, lock_fight_strikes=fight, fault=fault)
