"""RealGimbal torque-stub tests (no serial PTU, no pyserial)."""

from flight.hal.drivers_real import RealGimbal
from flight.libs.config import GimbalConfig
from flight.libs.time import ManualClock
from flight.libs.types import Ok


def test_set_torque_is_ok_noop() -> None:
    """set_torque returns Ok and does not require a vendor SDK."""
    gimbal = RealGimbal(clock=ManualClock(), cfg=GimbalConfig())
    assert isinstance(gimbal.set_torque(0.5), Ok)


def test_goto_angle_records_clamped_elevation() -> None:
    """goto_angle stores a travel-clamped elevation for later reads."""
    clock = ManualClock()
    gimbal = RealGimbal(clock=clock, cfg=GimbalConfig())
    assert isinstance(gimbal.goto_angle(500.0), Ok)
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    assert pos.value.el_deg == 90.0
    assert pos.value.timestamp_s == clock.monotonic_s()
    assert not hasattr(pos.value, "az_deg")


def test_stow_records_pose_and_switch() -> None:
    """stow() records the stow elevation and reports the switch True."""
    gimbal = RealGimbal(clock=ManualClock(), cfg=GimbalConfig())
    assert isinstance(gimbal.stow(), Ok)
    switch = gimbal.read_stow_switch()
    assert isinstance(switch, Ok)
    assert switch.value is True
