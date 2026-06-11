"""Simulated gimbal with first-order dynamics, travel/slew limits, and encoder noise.

Position integrates lazily: every public call first advances the internal state by the
clock time elapsed since the previous call, so the same driver is honest under the
threaded flight loop (RealClock) and the stepped SIL (ManualClock). ABSOLUTE/STOW/HOME
approach their target with a first-order exponential response clamped to the hardware
slew envelope; RATE integrates the clamped commanded rates. Position is clamped to the
travel limits after every update. Encoder reads add seeded Gaussian noise and carry
the monotonic read timestamp. send_command (delta) is retained temporarily for the
legacy path and is removed by the pointing switchover.

Satisfies: REQ-AIML-GIMB-001, REQ-GIMB-HIGH-002.
"""

from __future__ import annotations

# stdlib
import math

# third-party
import numpy as np

# internal
from flight.hal.interfaces.gimbal import GimbalPosition
from flight.libs.config import GimbalConfig
from flight.libs.messages import GimbalCommandMsg
from flight.libs.time import Clock
from flight.libs.types import FaultCode, GimbalCommandMode, Ok, Result

_STOW_TOLERANCE_DEG = 0.5  # switch closes within this of the stow pose


class SimGimbal:
    """Gimbal driver with first-order dynamics for SIL (satisfies GimbalActuator).

    Attributes (internal):
        _clock: Injected time source; used to measure elapsed time between calls.
        _cfg: GimbalConfig (dynamics, limits, poses, noise).
        _az: Current azimuth pose (degrees).
        _el: Current elevation pose (degrees).
        _mode: Active command mode; None means no command issued yet.
        _target_az: Target azimuth for ABSOLUTE/STOW/HOME modes.
        _target_el: Target elevation for ABSOLUTE/STOW/HOME modes.
        _rate_az: Commanded azimuth rate (deg/s) for RATE mode.
        _rate_el: Commanded elevation rate (deg/s) for RATE mode.
        _stow_commanded: True once stow() has been called; enables stow-switch logic.
        _last_t: Monotonic time at the last _integrate() call.
        _rng: Seeded numpy Generator for reproducible encoder noise.
    """

    def __init__(
        self,
        clock: Clock,
        cfg: GimbalConfig | None = None,
        az_deg: float = 0.0,
        el_deg: float = 0.0,
    ) -> None:
        """Start at a pose with the configured dynamics and a seeded noise RNG.

        Args:
            clock: Injected time source for lazy integration.
            cfg: GimbalConfig; defaults to GimbalConfig() if None.
            az_deg: Initial azimuth in degrees.
            el_deg: Initial elevation in degrees.
        """
        self._clock = clock
        self._cfg = cfg if cfg is not None else GimbalConfig()
        self._az = az_deg
        self._el = el_deg
        self._mode: GimbalCommandMode | None = None
        self._target_az = az_deg
        self._target_el = el_deg
        self._rate_az = 0.0
        self._rate_el = 0.0
        self._stow_commanded = False
        self._last_t = clock.monotonic_s()
        self._rng = np.random.default_rng(self._cfg.sim_seed)

    def _clamp_travel(self) -> None:
        """Clamp the integrated pose into the configured travel limits."""
        cfg = self._cfg
        self._az = min(max(self._az, cfg.az_min_deg), cfg.az_max_deg)
        self._el = min(max(self._el, cfg.el_min_deg), cfg.el_max_deg)

    def _integrate(self) -> None:
        """Advance the pose by the clock time elapsed since the last call.

        Notes:
            RATE mode: integrates clamped commanded rates.
            ABSOLUTE/STOW/HOME modes: first-order exponential approach toward the
            target, clamped to the hardware slew envelope per step.
            No-op when dt <= 0 (repeated calls at the same clock time are idempotent).
        """
        now = self._clock.monotonic_s()
        dt = now - self._last_t
        self._last_t = now
        if dt <= 0.0:
            return
        cfg = self._cfg
        max_step = cfg.max_hw_slew_rate_deg_per_s * dt
        if self._mode is GimbalCommandMode.RATE:
            self._az += min(max(self._rate_az * dt, -max_step), max_step)
            self._el += min(max(self._rate_el * dt, -max_step), max_step)
        elif self._mode is not None:
            alpha = 1.0 - math.exp(-dt / cfg.sim_time_constant_s)
            az_step = (self._target_az - self._az) * alpha
            el_step = (self._target_el - self._el) * alpha
            self._az += min(max(az_step, -max_step), max_step)
            self._el += min(max(el_step, -max_step), max_step)
        self._clamp_travel()

    def goto_angle(self, az_deg: float, el_deg: float) -> Result[None, FaultCode]:
        """Set an absolute target, clamped into the travel limits.

        Args:
            az_deg: Target azimuth in degrees.
            el_deg: Target elevation in degrees.

        Returns:
            Ok(None) always (the sim never fails hardware commands).
        """
        self._integrate()
        cfg = self._cfg
        self._target_az = min(max(az_deg, cfg.az_min_deg), cfg.az_max_deg)
        self._target_el = min(max(el_deg, cfg.el_min_deg), cfg.el_max_deg)
        self._mode = GimbalCommandMode.ABSOLUTE
        self._stow_commanded = False
        return Ok(None)

    def set_rate(
        self, az_rate_deg_per_s: float, el_rate_deg_per_s: float
    ) -> Result[None, FaultCode]:
        """Set axis rates, clamped to the hardware slew envelope.

        Args:
            az_rate_deg_per_s: Azimuth rate in deg/s.
            el_rate_deg_per_s: Elevation rate in deg/s.

        Returns:
            Ok(None) always.
        """
        self._integrate()
        limit = self._cfg.max_hw_slew_rate_deg_per_s
        self._rate_az = min(max(az_rate_deg_per_s, -limit), limit)
        self._rate_el = min(max(el_rate_deg_per_s, -limit), limit)
        self._mode = GimbalCommandMode.RATE
        self._stow_commanded = False
        return Ok(None)

    def home(self) -> Result[None, FaultCode]:
        """Drive to the configured home pose.

        Returns:
            Ok(None) always.
        """
        self._integrate()
        self._target_az, self._target_el = self._cfg.home_az_deg, self._cfg.home_el_deg
        self._mode = GimbalCommandMode.HOME
        self._stow_commanded = False
        return Ok(None)

    def stow(self) -> Result[None, FaultCode]:
        """Drive to the configured stow pose and arm the stow switch.

        Returns:
            Ok(None) always.
        """
        self._integrate()
        self._target_az, self._target_el = self._cfg.stow_az_deg, self._cfg.stow_el_deg
        self._mode = GimbalCommandMode.STOW
        self._stow_commanded = True
        return Ok(None)

    def read_position(self) -> Result[GimbalPosition, FaultCode]:
        """Return the noisy, timestamped encoder pose.

        Returns:
            Ok(GimbalPosition) with Gaussian noise applied and the clock timestamp.
        """
        self._integrate()
        noise = self._rng.normal(0.0, self._cfg.sim_encoder_noise_deg, 2)
        return Ok(
            GimbalPosition(
                az_deg=self._az + float(noise[0]),
                el_deg=self._el + float(noise[1]),
                timestamp_s=self._last_t,
            )
        )

    def read_stow_switch(self) -> Result[bool, FaultCode]:
        """True once stow was commanded and the pose is within the switch tolerance.

        Returns:
            Ok(bool): True when stow was commanded and the gimbal is near the stow pose.
        """
        self._integrate()
        at_pose = (
            abs(self._az - self._cfg.stow_az_deg) < _STOW_TOLERANCE_DEG
            and abs(self._el - self._cfg.stow_el_deg) < _STOW_TOLERANCE_DEG
        )
        return Ok(self._stow_commanded and at_pose)

    def send_command(self, command: GimbalCommandMsg) -> Result[None, FaultCode]:
        """DEPRECATED legacy delta path (removed by the pointing switchover).

        Args:
            command: Legacy GimbalCommandMsg with az_delta_deg/el_delta_deg.

        Returns:
            Ok(None) always.
        """
        self._integrate()
        self._az += command.az_delta_deg
        self._el += command.el_delta_deg
        self._clamp_travel()
        return Ok(None)
