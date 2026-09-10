"""Tests for the light pointing-integrity detector."""

import math

from flight.libs.config import IntegrityConfig
from flight.libs.types import FaultCode
from flight.payload.gimbal.integrity import check_integrity, lock_hold_rate


def test_nan_trips_immediately() -> None:
    """A non-finite rate or torque is GIMBAL_RUNAWAY on the first check."""
    cfg = IntegrityConfig()
    result = check_integrity(cfg, math.nan, 0.0, 0.0, 0.0, False, 0, 0)
    assert result.fault is FaultCode.GIMBAL_RUNAWAY


def test_encoder_freeze_trips_after_strikes() -> None:
    """Nonzero r with a frozen encoder accumulates strikes to GIMBAL_RUNAWAY."""
    cfg = IntegrityConfig(freeze_strikes=3, r_min_rad_s=0.01, encoder_rate_ratio=0.2)
    freeze = 0
    fight = 0
    fault = None
    for _ in range(3):
        out = check_integrity(cfg, 0.05, 0.05, 0.1, 0.0, False, freeze, fight)
        freeze = out.freeze_strikes
        fight = out.lock_fight_strikes
        fault = out.fault
    assert fault is FaultCode.GIMBAL_RUNAWAY
    assert freeze == 3


def test_lock_fight_trips_after_strikes() -> None:
    """Lock engaged with hold-pose motion above the pin threshold trips after N strikes."""
    cfg = IntegrityConfig(lock_fight_strikes=2, lock_fight_rad_s=0.005)
    freeze = 0
    fight = 0
    fault = None
    for _ in range(2):
        out = check_integrity(cfg, 0.0, 0.02, 0.0, 0.02, True, freeze, fight, 0.02)
        freeze = out.freeze_strikes
        fight = out.lock_fight_strikes
        fault = out.fault
    assert fault is FaultCode.GIMBAL_RUNAWAY


def test_lock_fight_ignores_noisy_encoder_rate() -> None:
    """A large 1 kHz y_m at rest does not accumulate lock-fight strikes."""
    cfg = IntegrityConfig(lock_fight_strikes=2, lock_fight_rad_s=0.005)
    out = check_integrity(cfg, 0.0, 0.09, 0.0, 0.09, True, 0, 0, 0.0)
    assert out.lock_fight_strikes == 0
    assert out.fault is None


def test_lock_hold_rate_latches_then_measures_creep() -> None:
    """The first locked sample latches a pose; later displacement is a rate."""
    motion, ref_th, ref_t = lock_hold_rate(True, 0.10, 1.0, None, None)
    assert motion == 0.0
    assert ref_th == 0.10
    assert ref_t == 1.0
    creep, _, _ = lock_hold_rate(True, 0.12, 2.0, ref_th, ref_t)
    assert abs(creep - 0.02) < 1e-12
    open_motion, open_th, open_t = lock_hold_rate(False, 0.12, 3.0, ref_th, ref_t)
    assert open_motion == 0.0
    assert open_th is None
    assert open_t is None


def test_healthy_motion_clears_strikes() -> None:
    """Encoder rate that matches r resets the freeze counter."""
    cfg = IntegrityConfig(freeze_strikes=3, r_min_rad_s=0.01, encoder_rate_ratio=0.2)
    first = check_integrity(cfg, 0.05, 0.05, 0.1, 0.0, False, 0, 0)
    assert first.freeze_strikes == 1
    cleared = check_integrity(cfg, 0.05, 0.05, 0.1, 0.05, False, first.freeze_strikes, 0)
    assert cleared.freeze_strikes == 0
    assert cleared.fault is None
