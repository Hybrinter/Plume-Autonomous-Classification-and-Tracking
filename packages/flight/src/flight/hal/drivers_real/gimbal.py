"""Real two-axis PTU gimbal driver over a serial ASCII protocol (reference: FLIR PTU
E46-class; spec Section 2).

pyserial imports lazily in __init__ (the SDK-free CI pattern). Protocol subset, line
oriented: commands are '<verb><signed counts>\\n' writes; every command yields one
response line -- '*' prefix = success, '!' prefix = error. Angle <-> count conversion
uses GimbalConfig.counts_per_deg. Verbs: PP/TP = pan/tilt absolute position command
or (bare) position query; PS/TS = pan/tilt rate. The exact verb set is a documented
reference assumption to be validated at HIL bring-up against the actual unit's manual.
The driver enforces the travel and slew envelopes by clamping before conversion
(defense in depth below the arbiter's mission limits). A lock serializes transactions
(capture loop vs control plane). Driver-level failures map to GIMBAL_FAULT.

Satisfies: REQ-AIML-GIMB-001, REQ-GIMB-HIGH-004.
"""

from __future__ import annotations

# stdlib
import threading

# internal
from flight.hal.interfaces.gimbal import GimbalPosition
from flight.libs.config import GimbalConfig
from flight.libs.time import Clock
from flight.libs.types import Err, FaultCode, Ok, Result


class RealGimbal:
    """Serial PTU driver satisfying GimbalActuator structurally.

    Notes:
        The PTU exposes no discrete stow switch; read_stow_switch infers stow from the
        encoder pose (within 0.5 deg of the configured stow pose on both axes).
    """

    def __init__(
        self,
        clock: Clock,
        cfg: GimbalConfig | None = None,
        timeout_s: float = 1.0,
    ) -> None:
        """Open the configured serial port.

        Inputs:
            clock (Clock): Injected clock used to timestamp encoder reads.
            cfg (GimbalConfig | None): Gimbal envelope/pose/link config; None uses
                the GimbalConfig defaults.
            timeout_s (float): Serial read timeout in seconds (default 1.0).

        Raises:
            ImportError: If pyserial is not installed.
            ValueError: If cfg.serial_port is empty (startup misconfiguration).
        """
        try:
            import serial
        except ImportError as exc:
            raise ImportError(
                "pyserial is not installed. Install it to use RealGimbal; use "
                "SimGimbal in tests and simulation."
            ) from exc
        self._cfg = cfg if cfg is not None else GimbalConfig()
        if not self._cfg.serial_port:
            raise ValueError("GimbalConfig.serial_port must be set to use RealGimbal")
        self._serial_exc = serial.SerialException
        self._port = serial.Serial(
            port=self._cfg.serial_port, baudrate=self._cfg.serial_baud, timeout=timeout_s
        )
        self._clock = clock
        self._lock = threading.Lock()

    def _transact(self, command: str) -> Result[str, FaultCode]:
        """Write one command line and read its response.

        Inputs:
            command (str): The verb (+ optional signed counts) to send, without newline.

        Outputs:
            Result[str, FaultCode]: Ok(response line) on a '*' response; Err(GIMBAL_FAULT)
            on a '!' (or any non-'*') response or a serial I/O error.
        """
        try:
            self._port.write(f"{command}\n".encode("ascii"))
            response = self._port.readline().decode("ascii", errors="replace").strip()
        except self._serial_exc:
            return Err(FaultCode.GIMBAL_FAULT)
        if not response.startswith("*"):
            return Err(FaultCode.GIMBAL_FAULT)
        return Ok(response)

    def _counts(self, deg: float) -> int:
        """Convert degrees to encoder counts via GimbalConfig.counts_per_deg.

        Inputs:
            deg (float): Angle or rate in degrees (per second for rates).

        Outputs:
            int: The rounded encoder-count value.
        """
        return round(deg * self._cfg.counts_per_deg)

    def goto_angle(self, az_deg: float, el_deg: float) -> Result[None, FaultCode]:
        """Command absolute pan/tilt positions, clamped to the travel envelope.

        Inputs:
            az_deg (float): Target azimuth in degrees (clamped to az_min/az_max).
            el_deg (float): Target elevation in degrees (clamped to el_min/el_max).

        Outputs:
            Result[None, FaultCode]: Ok(None), or Err(GIMBAL_FAULT) on a PTU error.
        """
        cfg = self._cfg
        az = min(max(az_deg, cfg.az_min_deg), cfg.az_max_deg)
        el = min(max(el_deg, cfg.el_min_deg), cfg.el_max_deg)
        with self._lock:
            for verb, value in (("PP", az), ("TP", el)):
                result = self._transact(f"{verb}{self._counts(value)}")
                if isinstance(result, Err):
                    return Err(result.error)
        return Ok(None)

    def set_rate(
        self, az_rate_deg_per_s: float, el_rate_deg_per_s: float
    ) -> Result[None, FaultCode]:
        """Command pan/tilt rates, clamped to the hardware slew envelope.

        Inputs:
            az_rate_deg_per_s (float): Azimuth rate in deg/s (clamped to +-max_hw_slew).
            el_rate_deg_per_s (float): Elevation rate in deg/s (clamped to +-max_hw_slew).

        Outputs:
            Result[None, FaultCode]: Ok(None), or Err(GIMBAL_FAULT) on a PTU error.
        """
        limit = self._cfg.max_hw_slew_rate_deg_per_s
        az = min(max(az_rate_deg_per_s, -limit), limit)
        el = min(max(el_rate_deg_per_s, -limit), limit)
        with self._lock:
            for verb, value in (("PS", az), ("TS", el)):
                result = self._transact(f"{verb}{self._counts(value)}")
                if isinstance(result, Err):
                    return Err(result.error)
        return Ok(None)

    def home(self) -> Result[None, FaultCode]:
        """Drive to the configured home pose.

        Outputs:
            Result[None, FaultCode]: Ok(None), or Err(GIMBAL_FAULT) on a PTU error.
        """
        return self.goto_angle(self._cfg.home_az_deg, self._cfg.home_el_deg)

    def stow(self) -> Result[None, FaultCode]:
        """Drive to the configured stow pose.

        Outputs:
            Result[None, FaultCode]: Ok(None), or Err(GIMBAL_FAULT) on a PTU error.
        """
        return self.goto_angle(self._cfg.stow_az_deg, self._cfg.stow_el_deg)

    def read_position(self) -> Result[GimbalPosition, FaultCode]:
        """Query pan/tilt positions and convert counts to timestamped degrees.

        Outputs:
            Result[GimbalPosition, FaultCode]: Ok with the encoder pose stamped from the
            injected clock, or Err(GIMBAL_FAULT) on a PTU error or unparseable response.
        """
        with self._lock:
            counts: list[int] = []
            for verb in ("PP", "TP"):
                result = self._transact(verb)
                if isinstance(result, Err):
                    return Err(result.error)
                try:
                    counts.append(int(result.value.lstrip("* ").strip()))
                except ValueError:
                    return Err(FaultCode.GIMBAL_FAULT)
        return Ok(
            GimbalPosition(
                az_deg=counts[0] / self._cfg.counts_per_deg,
                el_deg=counts[1] / self._cfg.counts_per_deg,
                timestamp_s=self._clock.monotonic_s(),
            )
        )

    def read_stow_switch(self) -> Result[bool, FaultCode]:
        """Infer stow from encoder pose (the reference PTU exposes no discrete switch).

        Outputs:
            Result[bool, FaultCode]: Ok(True) when both axes are within 0.5 deg of the
            configured stow pose, Ok(False) otherwise, or Err(GIMBAL_FAULT) on a read
            failure.
        """
        pos = self.read_position()
        if isinstance(pos, Err):
            return Err(pos.error)
        return Ok(
            abs(pos.value.az_deg - self._cfg.stow_az_deg) < 0.5
            and abs(pos.value.el_deg - self._cfg.stow_el_deg) < 0.5
        )
