"""Simulated gimbal actuator.

Integrates azimuth/elevation delta commands into a software-tracked position.
Satisfies the GimbalActuator protocol. Later, SIL can replace this with a driver
backed by the dynamics model in sim/twin.
"""

from flight.hal.interfaces.gimbal import GimbalPosition
from flight.libs.messages import GimbalCommandMsg
from flight.libs.types import FaultCode, Ok, Result


class SimGimbal:
    """Gimbal actuator that accumulates az/el deltas in software (sim/SIL driver)."""

    def __init__(self, az_deg: float = 0.0, el_deg: float = 0.0) -> None:
        """Initialize at a starting pointing.

        Args:
            az_deg: Initial azimuth in degrees.
            el_deg: Initial elevation in degrees.
        """
        self._az_deg = az_deg
        self._el_deg = el_deg

    def send_command(self, command: GimbalCommandMsg) -> Result[None, FaultCode]:
        """Apply the command's az/el deltas to the tracked position."""
        self._az_deg += command.az_delta_deg
        self._el_deg += command.el_delta_deg
        return Ok(None)

    def read_position(self) -> Result[GimbalPosition, FaultCode]:
        """Return the current tracked az/el position."""
        return Ok(GimbalPosition(az_deg=self._az_deg, el_deg=self._el_deg))
