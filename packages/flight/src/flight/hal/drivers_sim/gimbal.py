"""Simulated elevation gimbal: rigid-body ODE plant, encoder quantization, noise.

Integrates J * omega_dot + B * omega = tau in SI. Lazy clock integration advances
the plant by elapsed monotonic time. Repeated `set_torque` at a frozen clock (SIL
catch-up) steps one inner period per call and records a catch-up debt so a later
clock jump does not double-count.

Satisfies: REQ-AIML-GIMB-001, REQ-GIMB-HIGH-002.
"""

from __future__ import annotations

# stdlib
import math

# third-party
import numpy as np

# internal
from flight.hal.interfaces.gimbal import GimbalHealth, GimbalPosition
from flight.libs.config import GimbalConfig
from flight.libs.time import Clock
from flight.libs.types import Err, FaultCode, Ok, Result

_STOW_TOLERANCE_DEG = 0.5


class SimGimbal:
    """Elevation plant with viscous-stand-in damping (satisfies GimbalActuator).

    Attributes (internal):
        _clock: Injected time source.
        _cfg: GimbalConfig (plant, limits, encoder, noise).
        _theta_rad: True elevation, rad.
        _omega_rad_s: True rate, rad/s.
        _tau_nm: Held torque command, N·m.
        _stow_commanded: True once stow() has been called.
        _last_t: Monotonic time at the last clock-based integrate.
        _catchup_debt_s: Plant time already applied at a frozen clock.
        _inner_dt_s: Catch-up inner period when the clock does not advance.
        _rng: Seeded numpy Generator for encoder noise.
        _last_feedback_s: Timestamp of the last encoder sample, or None.
    """

    def __init__(
        self,
        clock: Clock,
        cfg: GimbalConfig | None = None,
        el_deg: float = 0.0,
        inner_dt_s: float = 0.001,
    ) -> None:
        """Start at an elevation with the configured plant and a seeded noise RNG.

        Args:
            clock: Injected time source for lazy integration.
            cfg: GimbalConfig; defaults to GimbalConfig() if None.
            el_deg: Initial elevation in degrees.
            inner_dt_s: Inner period used for frozen-clock catch-up steps.
        """
        self._clock = clock
        self._cfg = cfg if cfg is not None else GimbalConfig()
        self._theta_rad = math.radians(el_deg)
        self._omega_rad_s = 0.0
        self._tau_nm = 0.0
        self._target_el_deg = el_deg
        self._stow_commanded = False
        self._last_t = clock.monotonic_s()
        self._catchup_debt_s = 0.0
        self._inner_dt_s = inner_dt_s
        self._rng = np.random.default_rng(self._cfg.sim_seed)
        self._encoder_frozen = False
        self._frozen_el_deg = math.degrees(self._theta_rad)
        self._command_valid_until_s: float | None = None
        self._inhibited = True
        self._last_feedback_s: float | None = None

    def _expire_command_if_needed(self, now: float) -> None:
        """Fail closed when the host has not refreshed command authority."""
        if self._command_valid_until_s is not None and now >= self._command_valid_until_s:
            self._tau_nm = 0.0
            self._inhibited = True

    def _clip_tau(self, tau_nm: float) -> float:
        """Clip torque to +-tau_max_nm."""
        limit = self._cfg.tau_max_nm
        return min(max(tau_nm, -limit), limit)

    def _ode_step(self, dt_s: float) -> None:
        """Advance the rigid-body plant by dt_s with the held torque.

        Args:
            dt_s: Integration step, seconds. Non-positive is a no-op.
        """
        if dt_s <= 0.0:
            return
        cfg = self._cfg
        j = cfg.J_kg_m2
        b = cfg.B_nms_per_rad
        tau = self._tau_nm
        omega_max = math.radians(cfg.max_hw_slew_rate_deg_per_s)
        theta_min = math.radians(cfg.el_hw_min_deg)
        theta_max = math.radians(cfg.el_hw_max_deg)
        # Semi-implicit Euler on J * omega_dot + B * omega = tau
        omega_dot = (tau - b * self._omega_rad_s) / j
        omega = self._omega_rad_s + omega_dot * dt_s
        omega = min(max(omega, -omega_max), omega_max)
        theta = self._theta_rad + omega * dt_s
        if theta > theta_max:
            theta = theta_max
            if omega > 0.0:
                omega = 0.0
        elif theta < theta_min:
            theta = theta_min
            if omega < 0.0:
                omega = 0.0
        self._theta_rad = theta
        self._omega_rad_s = omega

    def _integrate_clock(self) -> None:
        """Advance the plant by elapsed clock time, minus catch-up debt.

        A frozen clock (dt ~ 0) leaves catch-up debt in place. Clearing debt on a
        frozen read would make a later clock jump double-count plant time.
        """
        now = self._clock.monotonic_s()
        self._expire_command_if_needed(now)
        dt = now - self._last_t
        if dt > 1e-12:
            dt_eff = dt - self._catchup_debt_s
            self._catchup_debt_s = 0.0
            self._last_t = now
            if dt_eff > 0.0:
                self._ode_step(dt_eff)

    def set_torque(
        self, tau_nm: float, valid_until_s: float | None = None
    ) -> Result[None, FaultCode]:
        """Hold a clipped torque. Integrate clock dt, or one inner period if frozen.

        Args:
            tau_nm: Commanded torque, N·m.

        Returns:
            Ok(None) always.
        """
        now = self._clock.monotonic_s()
        if not math.isfinite(tau_nm):
            self._tau_nm = 0.0
            self._inhibited = True
            return Err(FaultCode.GIMBAL_FAULT)
        if valid_until_s is not None and valid_until_s <= now:
            self._tau_nm = 0.0
            self._inhibited = True
            return Err(FaultCode.GIMBAL_FAULT)
        dt = now - self._last_t
        if dt > 1e-12:
            dt_eff = dt - self._catchup_debt_s
            self._catchup_debt_s = 0.0
            self._last_t = now
            if dt_eff > 0.0:
                self._ode_step(dt_eff)
            self._tau_nm = self._clip_tau(tau_nm)
        else:
            self._tau_nm = self._clip_tau(tau_nm)
            self._ode_step(self._inner_dt_s)
            self._catchup_debt_s += self._inner_dt_s
        # A legacy caller that omits an authority lease cannot leave torque held
        # indefinitely; the simulated hardware expires it after one inner period.
        self._command_valid_until_s = (
            now + self._inner_dt_s if valid_until_s is None else valid_until_s
        )
        self._inhibited = tau_nm == 0.0
        return Ok(None)

    def inhibit(self, reason: str) -> Result[GimbalHealth, FaultCode]:
        """Immediately de-energize the simulated drive and report confirmation."""
        del reason
        self._integrate_clock()
        self._tau_nm = 0.0
        self._command_valid_until_s = self._clock.monotonic_s()
        self._inhibited = True
        return Ok(self._health())

    def _health(self) -> GimbalHealth:
        """Build the driver's current local health evidence."""
        return GimbalHealth(
            feedback_valid=True,
            last_feedback_s=self._last_feedback_s,
            command_valid_until_s=self._command_valid_until_s,
            inhibited=self._inhibited,
            inhibit_confirmed=self._inhibited,
        )

    def read_health(self) -> Result[GimbalHealth, FaultCode]:
        """Return driver-side command-expiry and feedback evidence."""
        self._integrate_clock()
        return Ok(self._health())

    def goto_angle(self, el_deg: float) -> Result[None, FaultCode]:
        """Record a position-loop target. Motion comes from torque, not this call.

        Args:
            el_deg: Target elevation in degrees.

        Returns:
            Ok(None) always.
        """
        self._integrate_clock()
        cfg = self._cfg
        self._target_el_deg = min(max(el_deg, cfg.el_hw_min_deg), cfg.el_hw_max_deg)
        self._stow_commanded = False
        return Ok(None)

    def home(self) -> Result[None, FaultCode]:
        """Record the configured home pose as the position-loop target.

        Returns:
            Ok(None) always.
        """
        return self.goto_angle(self._cfg.home_el_deg)

    def stow(self) -> Result[None, FaultCode]:
        """Record the stow pose and arm the stow switch.

        Returns:
            Ok(None) always.
        """
        self._integrate_clock()
        self._target_el_deg = self._cfg.stow_el_deg
        self._stow_commanded = True
        return Ok(None)

    def _quantize_deg(self, theta_rad: float) -> float:
        """Quantize true elevation to encoder counts, then add Gaussian noise.

        Args:
            theta_rad: True elevation, rad.

        Returns:
            Noisy encoder elevation, degrees.
        """
        cpr = float(self._cfg.encoder_counts_per_rev)
        theta_deg = math.degrees(theta_rad)
        counts = round(theta_deg / 360.0 * cpr)
        quantized = counts / cpr * 360.0
        noise = float(self._rng.normal(0.0, self._cfg.sim_encoder_noise_deg))
        return quantized + noise

    def freeze_encoder(self) -> None:
        """Hold encoder reads at the current quantized pose (integrity injection)."""
        self._integrate_clock()
        self._encoder_frozen = True
        self._frozen_el_deg = self._quantize_deg(self._theta_rad)

    def advance_plant(self) -> None:
        """Integrate the plant to the clock without taking an encoder sample.

        Not on GimbalActuator. SIL bind uses this for true shutter pose. It does
        not consume encoder-noise RNG or update last_feedback_s.

        Returns:
            None.
        """
        self._integrate_clock()

    def read_position(self) -> Result[GimbalPosition, FaultCode]:
        """Return the quantized, noisy, timestamped encoder elevation.

        Returns:
            Ok(GimbalPosition) with the clock timestamp. Frozen when freeze_encoder ran.
        """
        self._integrate_clock()
        if self._encoder_frozen:
            el_deg = self._frozen_el_deg
        else:
            el_deg = self._quantize_deg(self._theta_rad)
        # Deterministic catch-up calls advance the simulated plant while a
        # ManualClock is frozen. Stamp that virtual plant time so the flight
        # rate fit sees the same chronology as the simulated encoder.
        sample_t_s = self._last_t + self._catchup_debt_s
        self._last_feedback_s = sample_t_s
        return Ok(
            GimbalPosition(
                el_deg=el_deg,
                timestamp_s=sample_t_s,
            )
        )

    def read_stow_switch(self) -> Result[bool, FaultCode]:
        """True once stow was commanded and elevation is within the switch tolerance.

        Returns:
            Ok(bool): True when stow was commanded and the gimbal is near stow.
        """
        self._integrate_clock()
        at_pose = abs(math.degrees(self._theta_rad) - self._cfg.stow_el_deg) < _STOW_TOLERANCE_DEG
        return Ok(self._stow_commanded and at_pose)

    @property
    def true_el_deg(self) -> float:
        """True (pre-encoder) elevation in degrees. Observability for analysis tools."""
        return math.degrees(self._theta_rad)

    @property
    def true_omega_rad_s(self) -> float:
        """True plant rate in rad/s. Observability for analysis tools."""
        return self._omega_rad_s
