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


def test_goto_angle_approaches_elevation_target_with_lag() -> None:
    """An absolute command moves elevation toward the target; azimuth stays pinned at 0."""
    clock = ManualClock()
    gimbal = _gimbal(clock)
    assert isinstance(gimbal.goto_angle(10.0, 20.0), Ok)
    clock.advance(0.1)
    mid = gimbal.read_position()
    assert isinstance(mid, Ok)
    assert abs(mid.value.az_deg) < 1e-9
    assert 0.0 < mid.value.el_deg < 20.0
    clock.advance(30.0)
    settled = gimbal.read_position()
    assert isinstance(settled, Ok)
    assert abs(settled.value.az_deg) < 1e-9
    assert abs(settled.value.el_deg - 20.0) < 0.1


def test_slew_rate_is_limited() -> None:
    """Motion toward a far elevation target never exceeds the hardware slew envelope."""
    clock = ManualClock()
    gimbal = _gimbal(clock, max_hw_slew_rate_deg_per_s=10.0, sim_time_constant_s=0.001)
    gimbal.goto_angle(0.0, 90.0)
    clock.advance(1.0)
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    assert pos.value.el_deg <= 10.0 + 1e-6
    assert abs(pos.value.az_deg) < 1e-9


def test_set_rate_integrates_and_clamps_travel() -> None:
    """Elevation rate commands integrate position and stop at the hardware travel limit."""
    clock = ManualClock()
    gimbal = _gimbal(clock)
    assert isinstance(gimbal.set_rate(5.0, 2.0), Ok)
    clock.advance(1.0)
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    assert abs(pos.value.az_deg) < 1e-6
    assert abs(pos.value.el_deg - 2.0) < 1e-6
    clock.advance(100.0)
    clamped = gimbal.read_position()
    assert isinstance(clamped, Ok)
    assert clamped.value.el_deg == 90.0
    assert abs(clamped.value.az_deg) < 1e-6


def test_stow_reaches_pose_and_sets_switch() -> None:
    """stow() drives to the configured stow pose; the switch reads True on arrival."""
    clock = ManualClock()
    gimbal = _gimbal(clock)
    assert isinstance(gimbal.stow(), Ok)
    early = gimbal.read_stow_switch()
    assert isinstance(early, Ok)
    assert early.value is False
    clock.advance(60.0)
    done = gimbal.read_stow_switch()
    assert isinstance(done, Ok)
    assert done.value is True
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    assert abs(pos.value.el_deg - (-45.0)) < 0.5


def test_read_position_is_timestamped() -> None:
    """Encoder reads carry the monotonic read time."""
    clock = ManualClock()
    gimbal = _gimbal(clock)
    clock.advance(3.5)
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    assert pos.value.timestamp_s == clock.monotonic_s()
