"""Real gimbal driver stub. The PTU ASCII path is removed.

`set_torque` is a no-op `Ok` until the motor-amp interface exists. Pose methods
latch a commanded elevation for stow-switch arming. They do not close a position
or rate loop and they do not teleport the encoder. Encoder reads stay at the last
physical pose (0 until a future amp moves the axis). The stow switch is True only
when stow was commanded and the encoder is near stow — False on this stub until
hardware exists.

Satisfies: REQ-AIML-GIMB-001, REQ-GIMB-HIGH-004.
"""

from __future__ import annotations

from flight.hal.interfaces.gimbal import GimbalHealth, GimbalPosition
from flight.libs.config import GimbalConfig
from flight.libs.time import Clock
from flight.libs.types import Err, FaultCode, Ok, Result


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
        self._target_el_deg = 0.0
        self._stow_commanded = False

    def set_torque(
        self, tau_nm: float, valid_until_s: float | None = None
    ) -> Result[None, FaultCode]:
        """Refuse drive authority until a motor amplifier is implemented.

        Inputs:
            tau_nm (float): Commanded torque in N·m (ignored).

        Outputs:
            Err(GIMBAL_FAULT): this stub has no independent expiry or inhibit path.
        """
        del tau_nm, valid_until_s
        return Err(FaultCode.GIMBAL_FAULT)

    def inhibit(self, reason: str) -> Result[GimbalHealth, FaultCode]:
        """Refuse to claim containment without a wired amplifier inhibit channel."""
        del reason
        return Err(FaultCode.GIMBAL_FAULT)

    def read_health(self) -> Result[GimbalHealth, FaultCode]:
        """Report that this development stub has no verified containment evidence."""
        return Ok(
            GimbalHealth(
                feedback_valid=False,
                last_feedback_s=None,
                command_valid_until_s=None,
                inhibited=True,
                inhibit_confirmed=False,
            )
        )

    def goto_angle(self, el_deg: float) -> Result[None, FaultCode]:
        """Latch a travel-clamped pose target. Encoder is unchanged.

        Inputs:
            el_deg (float): Target elevation in degrees.

        Outputs:
            Ok(None).
        """
        cfg = self._cfg
        self._target_el_deg = min(max(el_deg, cfg.el_hw_min_deg), cfg.el_hw_max_deg)
        self._stow_commanded = False
        return Ok(None)

    def home(self) -> Result[None, FaultCode]:
        """Latch the configured home pose. Encoder is unchanged.

        Outputs:
            Ok(None).
        """
        return self.goto_angle(self._cfg.home_el_deg)

    def stow(self) -> Result[None, FaultCode]:
        """Latch the configured stow pose. Encoder is unchanged.

        Outputs:
            Ok(None).
        """
        result = self.goto_angle(self._cfg.stow_el_deg)
        self._stow_commanded = True
        return result

    def read_position(self) -> Result[GimbalPosition, FaultCode]:
        """Return the encoder pose (0 until a future amp moves it), timestamped.

        Outputs:
            Ok(GimbalPosition).
        """
        return Ok(GimbalPosition(el_deg=self._el_deg, timestamp_s=self._clock.monotonic_s()))

    def read_stow_switch(self) -> Result[bool, FaultCode]:
        """True when stow was commanded and the encoder is near stow.

        Outputs:
            Ok(bool). False on this stub because the encoder does not teleport.
        """
        at_pose = abs(self._el_deg - self._cfg.stow_el_deg) < 0.5
        return Ok(self._stow_commanded and at_pose)
