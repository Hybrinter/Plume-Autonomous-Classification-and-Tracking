"""Tests for SimGimbal first-order dynamics, limits, and the closed-loop HAL surface."""

from flight.hal.drivers_sim import SimGimbal
from flight.libs.config import GimbalConfig
from flight.libs.time import ManualClock
from flight.libs.types import Ok


def _gimbal(clock: ManualClock, **cfg_overrides: float) -> SimGimbal:
    """Construct a noiseless SimGimbal with optional GimbalConfig overrides."""
    # cfg_overrides only ever carries float fields; the ignore covers GimbalConfig's
    # heterogeneous (int/str) fields that the **splat cannot narrow.
    cfg = GimbalConfig(sim_encoder_noise_deg=0.0, **cfg_overrides)  # type: ignore[arg-type]
    return SimGimbal(clock=clock, cfg=cfg)


def test_set_position_approaches_target_with_lag() -> None:
    """An absolute command moves the gimbal toward the target, not instantly onto it."""
    clock = ManualClock()
    gimbal = _gimbal(clock)
    assert isinstance(gimbal.set_position(10.0), Ok)
    clock.advance(0.1)
    mid = gimbal.read_state()
    assert isinstance(mid, Ok)
    assert 0.0 < mid.value.position_deg < 10.0
    clock.advance(30.0)
    settled = gimbal.read_state()
    assert isinstance(settled, Ok)
    assert abs(settled.value.position_deg - 10.0) < 0.1


def test_slew_rate_is_limited() -> None:
    """Motion toward a far target never exceeds the hardware slew envelope."""
    clock = ManualClock()
    gimbal = _gimbal(clock, max_hw_slew_rate_deg_per_s=10.0, sim_time_constant_s=0.001)
    gimbal.set_position(90.0)
    clock.advance(1.0)
    pos = gimbal.read_state()
    assert isinstance(pos, Ok)
    assert pos.value.position_deg <= 10.0 + 1e-6


def test_set_velocity_integrates_and_clamps_travel() -> None:
    """Rate commands integrate position and stop at the travel limit."""
    clock = ManualClock()
    gimbal = _gimbal(clock, hardware_max_deg=5.0, operational_max_deg=5.0)
    assert isinstance(gimbal.set_velocity(2.0), Ok)
    clock.advance(1.0)
    pos = gimbal.read_state()
    assert isinstance(pos, Ok)
    assert abs(pos.value.position_deg - 2.0) < 1e-6
    clock.advance(10.0)
    clamped = gimbal.read_state()
    assert isinstance(clamped, Ok)
    assert clamped.value.position_deg == 5.0


def test_stow_reaches_pose_and_sets_switch() -> None:
    """stow() drives to the configured stow pose; the switch reads True on arrival."""
    clock = ManualClock()
    gimbal = _gimbal(clock)
    assert isinstance(gimbal.stow(), Ok)
    early = gimbal.read_state()
    assert isinstance(early, Ok)
    assert not early.value.position_reached
    clock.advance(60.0)
    done = gimbal.read_state()
    assert isinstance(done, Ok)
    assert done.value.position_reached
    pos = gimbal.read_state()
    assert isinstance(pos, Ok)
    assert abs(pos.value.position_deg - (-45.0)) < 0.5


def test_read_state_is_timestamped() -> None:
    """Encoder reads carry the monotonic read time."""
    clock = ManualClock()
    gimbal = _gimbal(clock)
    clock.advance(3.5)
    pos = gimbal.read_state()
    assert isinstance(pos, Ok)
    assert pos.value.timestamp_s == clock.monotonic_s()
