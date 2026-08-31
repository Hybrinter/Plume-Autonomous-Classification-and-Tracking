"""Simulated gimbal with inertia-plant torque dynamics, rate/position modes, and encoder noise.

Position integrates lazily: every public call first advances the internal state by the
clock time elapsed since the previous call, so the same driver is honest under the
threaded flight loop (RealClock) and the stepped SIL (ManualClock). Torque mode
integrates J ω̇ + B ω = τ in SI units (rad, N·m) and converts pose to degrees for the
encoder surface. ABSOLUTE/STOW/HOME approach their target with a first-order
exponential response clamped to the hardware slew envelope; RATE integrates the
clamped commanded rates. Position is clamped to the travel limits after every update;
an axis that hits a stop has its angular rate zeroed. Encoder reads add seeded
Gaussian noise and carry the monotonic read timestamp.

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
from flight.libs.time import Clock
from flight.libs.types import FaultCode, GimbalCommandMode, Ok, Result

_STOW_TOLERANCE_DEG = 0.5  # switch closes within this of the stow pose
_DEG_PER_RAD = 180.0 / math.pi
_RAD_PER_DEG = math.pi / 180.0


def _as_2x2(values: tuple[float, ...]) -> np.ndarray:  # np.ndarray[float64, (2, 2)]
    """Reshape a row-major length-4 tuple into a 2x2 matrix."""
    return np.array([[values[0], values[1]], [values[2], values[3]]], dtype=np.float64)


class SimGimbal:
    """Gimbal driver with inertia and first-order dynamics for SIL (satisfies GimbalActuator).

    Attributes (internal):
        _clock: Injected time source; used to measure elapsed time between calls.
        _cfg: GimbalConfig (dynamics, limits, poses, noise).
        _az: Current azimuth pose (degrees).
        _el: Current elevation pose (degrees).
        _omega: Angular rate in rad/s, shape (2,) for azimuth then elevation.
        _tau: Applied torque in N·m, shape (2,).
        _J_inv: Inverse inertia matrix (kg m^2)^-1.
        _B: Viscous damping matrix (N m s / rad).
        _torque_mode: True after set_torque until a rate or position command.
        _mode: Active command mode for rate/position paths; None until first command.
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
        self._omega = np.zeros(2, dtype=np.float64)  # np.ndarray[float64, (2,)]
        self._tau = np.zeros(2, dtype=np.float64)  # np.ndarray[float64, (2,)]
        inertia = _as_2x2(self._cfg.J_kg_m2)
        self._J_inv = np.linalg.inv(inertia)  # np.ndarray[float64, (2, 2)]
        self._B = _as_2x2(self._cfg.B_nms_per_rad)
        self._torque_mode = False
        self._mode: GimbalCommandMode | None = None
        self._target_az = az_deg
        self._target_el = el_deg
        self._rate_az = 0.0
        self._rate_el = 0.0
        self._stow_commanded = False
        self._last_t = clock.monotonic_s()
        self._rng = np.random.default_rng(self._cfg.sim_seed)

    def _clamp_travel(self) -> None:
        """Clamp the integrated pose into the configured travel limits and zero rate at a stop."""
        cfg = self._cfg
        if self._az <= cfg.az_min_deg:
            self._az = cfg.az_min_deg
            if self._omega[0] < 0.0:
                self._omega[0] = 0.0
        elif self._az >= cfg.az_max_deg:
            self._az = cfg.az_max_deg
            if self._omega[0] > 0.0:
                self._omega[0] = 0.0
        if self._el <= cfg.el_min_deg:
            self._el = cfg.el_min_deg
            if self._omega[1] < 0.0:
                self._omega[1] = 0.0
        elif self._el >= cfg.el_max_deg:
            self._el = cfg.el_max_deg
            if self._omega[1] > 0.0:
                self._omega[1] = 0.0

    def _integrate(self) -> None:
        """Advance the pose by the clock time elapsed since the last call.

        Notes:
            Torque mode: integrates J ω̇ + B ω = τ in SI, then converts to degrees.
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
        if self._torque_mode:
            residual = self._tau - self._B @ self._omega
            omega_dot = self._J_inv @ residual
            self._omega = self._omega + omega_dot * dt
            max_w = cfg.max_hw_slew_rate_deg_per_s * _RAD_PER_DEG
            self._omega = np.clip(self._omega, -max_w, max_w)
            self._az += float(self._omega[0]) * dt * _DEG_PER_RAD
            self._el += float(self._omega[1]) * dt * _DEG_PER_RAD
        elif self._mode is GimbalCommandMode.RATE:
            az_step = min(max(self._rate_az * dt, -max_step), max_step)
            el_step = min(max(self._rate_el * dt, -max_step), max_step)
            self._az += az_step
            self._el += el_step
            self._omega[0] = az_step / dt * _RAD_PER_DEG
            self._omega[1] = el_step / dt * _RAD_PER_DEG
        elif self._mode is not None:
            alpha = 1.0 - math.exp(-dt / cfg.sim_time_constant_s)
            az_step = min(max((self._target_az - self._az) * alpha, -max_step), max_step)
            el_step = min(max((self._target_el - self._el) * alpha, -max_step), max_step)
            self._az += az_step
            self._el += el_step
            self._omega[0] = az_step / dt * _RAD_PER_DEG
            self._omega[1] = el_step / dt * _RAD_PER_DEG
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
        self._torque_mode = False
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
        self._torque_mode = False
        self._stow_commanded = False
        return Ok(None)

    def set_torque(self, az_nm: float, el_nm: float) -> Result[None, FaultCode]:
        """Apply axis torques in newton-metres and switch to the inertia plant.

        Args:
            az_nm: Azimuth torque in N·m.
            el_nm: Elevation torque in N·m.

        Returns:
            Ok(None) always.
        """
        self._integrate()
        self._tau[0] = az_nm
        self._tau[1] = el_nm
        self._torque_mode = True
        self._mode = None
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
        self._torque_mode = False
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
        self._torque_mode = False
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
