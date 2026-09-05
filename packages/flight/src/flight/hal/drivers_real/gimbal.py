"""Real gimbal driver stub. The PTU ASCII path is removed.

`set_torque` and pose commands succeed as no-ops until the motor-amp interface
exists. Encoder reads return elevation 0. This stub does not import a vendor SDK.

Satisfies: REQ-AIML-GIMB-001, REQ-GIMB-HIGH-004.
"""

from __future__ import annotations

from flight.hal.interfaces.gimbal import GimbalPosition
from flight.libs.config import GimbalConfig
from flight.libs.time import Clock
from flight.libs.types import FaultCode, Ok, Result


class RealGimbal:
    """Torque-command stub satisfying GimbalActuator structurally.

    Notes:
        Amp current mapping (K_t) is not implemented. Commands return Ok and do
        not move hardware.
    """

    def __init__(
        self,
        clock: Clock,
        cfg: GimbalConfig | None = None,
    ) -> None:
        """Hold the clock and config. No serial port is opened.

        Inputs:
            clock (Clock): Injected clock used to timestamp encoder reads.
            cfg (GimbalConfig | None): Envelope config; None uses defaults.
        """
        self._cfg = cfg if cfg is not None else GimbalConfig()
        self._clock = clock
        self._el_deg = 0.0
        self._stow_commanded = False

    def set_torque(self, tau_nm: float) -> Result[None, FaultCode]:
        """Accept a torque command. The amp interface is not wired; this is a no-op.

        Inputs:
            tau_nm (float): Commanded torque in N·m (ignored).

        Outputs:
            Ok(None).
        """
        del tau_nm
        return Ok(None)

    def goto_angle(self, el_deg: float) -> Result[None, FaultCode]:
        """Record a pose target. No hardware motion.

        Inputs:
            el_deg (float): Target elevation in degrees.

        Outputs:
            Ok(None).
        """
        cfg = self._cfg
        self._el_deg = min(max(el_deg, cfg.el_hw_min_deg), cfg.el_hw_max_deg)
        self._stow_commanded = False
        return Ok(None)

    def home(self) -> Result[None, FaultCode]:
        """Record the configured home pose.

        Outputs:
            Ok(None).
        """
        return self.goto_angle(self._cfg.home_el_deg)

    def stow(self) -> Result[None, FaultCode]:
        """Record the configured stow pose.

        Outputs:
            Ok(None).
        """
        self._stow_commanded = True
        return self.goto_angle(self._cfg.stow_el_deg)

    def read_position(self) -> Result[GimbalPosition, FaultCode]:
        """Return the last recorded pose (0 until a pose command), timestamped.

        Outputs:
            Ok(GimbalPosition).
        """
        return Ok(GimbalPosition(el_deg=self._el_deg, timestamp_s=self._clock.monotonic_s()))

    def read_stow_switch(self) -> Result[bool, FaultCode]:
        """True when stow was commanded and the recorded pose is near stow.

        Outputs:
            Ok(bool).
        """
        at_pose = abs(self._el_deg - self._cfg.stow_el_deg) < 0.5
        return Ok(self._stow_commanded and at_pose)
