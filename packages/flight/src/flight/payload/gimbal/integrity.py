"""Light pointing-integrity detector (pure). NaN and encoder freeze.

Not the deleted RATE-mode runaway monitor. Strikes accumulate across inner ticks.
NaN torque or rate is immediate. Encoder freeze: commanded |r| above r_min while
the measured encoder rate is a small fraction of |r|.
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
        fault: GIMBAL_RUNAWAY when a trip fires, else None.
    """

    freeze_strikes: int
    fault: FaultCode | None


def check_integrity(
    cfg: IntegrityConfig,
    r_rad_s: float,
    y_m: float,
    tau_nm: float,
    encoder_rate_rad_s: float,
    freeze_strikes: int,
) -> IntegrityResult:
    """Advance strike counters and trip GIMBAL_RUNAWAY when a rule fires.

    Inputs:
        cfg: Integrity thresholds and strike limits.
        r_rad_s: Rate reference, rad/s.
        y_m: Encoder-rate estimate, rad/s.
        tau_nm: Torque command, N·m.
        encoder_rate_rad_s: Raw two-sample encoder rate, rad/s.
        freeze_strikes: Prior encoder-freeze count.

    Outputs:
        IntegrityResult: Updated counts and optional fault.
    """
    if not (math.isfinite(r_rad_s) and math.isfinite(y_m) and math.isfinite(tau_nm)):
        return IntegrityResult(
            freeze_strikes=freeze_strikes,
            fault=FaultCode.GIMBAL_RUNAWAY,
        )

    freeze = freeze_strikes
    if abs(r_rad_s) > cfg.r_min_rad_s and abs(encoder_rate_rad_s) < cfg.encoder_rate_ratio * abs(
        r_rad_s
    ):
        freeze += 1
    else:
        freeze = 0

    fault: FaultCode | None = None
    if freeze >= cfg.freeze_strikes:
        fault = FaultCode.GIMBAL_RUNAWAY
    return IntegrityResult(freeze_strikes=freeze, fault=fault)
