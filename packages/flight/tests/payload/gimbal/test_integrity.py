"""Tests for the light pointing-integrity detector."""

import math

from flight.libs.config import IntegrityConfig
from flight.libs.types import FaultCode
from flight.payload.gimbal.integrity import check_integrity


def test_nan_trips_immediately() -> None:
    """A non-finite rate or torque is GIMBAL_RUNAWAY on the first check."""
    cfg = IntegrityConfig()
    result = check_integrity(cfg, math.nan, 0.0, 0.0, 0.0, 0)
    assert result.fault is FaultCode.GIMBAL_RUNAWAY


def test_encoder_freeze_trips_after_strikes() -> None:
    """Nonzero r with a frozen encoder accumulates strikes to GIMBAL_RUNAWAY."""
    cfg = IntegrityConfig(freeze_strikes=3, r_min_rad_s=0.01, encoder_rate_ratio=0.2)
    freeze = 0
    fault = None
    for _ in range(3):
        out = check_integrity(cfg, 0.05, 0.05, 0.1, 0.0, freeze)
        freeze = out.freeze_strikes
        fault = out.fault
    assert fault is FaultCode.GIMBAL_RUNAWAY
    assert freeze == 3


def test_healthy_motion_clears_strikes() -> None:
    """Encoder rate that matches r resets the freeze counter."""
    cfg = IntegrityConfig(freeze_strikes=3, r_min_rad_s=0.01, encoder_rate_ratio=0.2)
    first = check_integrity(cfg, 0.05, 0.05, 0.1, 0.0, 0)
    assert first.freeze_strikes == 1
    cleared = check_integrity(cfg, 0.05, 0.05, 0.1, 0.05, first.freeze_strikes)
    assert cleared.freeze_strikes == 0
    assert cleared.fault is None
