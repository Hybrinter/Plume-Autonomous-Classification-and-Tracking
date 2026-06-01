"""Tests for the simulated gimbal actuator."""

from flight.hal.drivers_sim import SimGimbal
from flight.libs.messages import GimbalCommandMsg
from flight.libs.types import GimbalState, MessageType, Ok


def _command(az: float, el: float) -> GimbalCommandMsg:
    """Build a GimbalCommandMsg with the given az/el deltas."""
    return GimbalCommandMsg(
        msg_type=MessageType.GIMBAL_COMMAND,
        timestamp_utc="2026-05-31T00:00:00.000Z",
        frame_id=1,
        az_delta_deg=az,
        el_delta_deg=el,
        state=GimbalState.TRACKING,
        reason="test",
    )


def test_accumulates_position() -> None:
    """send_command integrates az/el deltas into the tracked position."""
    gimbal = SimGimbal()
    gimbal.send_command(_command(1.0, 2.0))
    gimbal.send_command(_command(0.5, -1.0))
    result = gimbal.read_position()
    assert isinstance(result, Ok)
    assert result.value.az_deg == 1.5
    assert result.value.el_deg == 1.0
