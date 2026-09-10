"""Deterministic single-axis elevation gimbal simulator."""

from __future__ import annotations

import math

import numpy as np

from flight.hal.interfaces.gimbal import GimbalAxisState
from flight.libs.config import GimbalConfig
from flight.libs.time import Clock
from flight.libs.types import FaultCode, GimbalCommandMode, Ok, Result

_STOW_TOLERANCE_DEG = 0.5


class SimGimbal:
    """One-axis first-order plant implementing :class:`GimbalActuator`."""

    def __init__(
        self,
        clock: Clock,
        cfg: GimbalConfig | None = None,
        position_deg: float = 0.0,
    ) -> None:
        self._clock = clock
        self._cfg = cfg if cfg is not None else GimbalConfig()
        self._position = position_deg
        self._target = self._position
        self._velocity = 0.0
        self._mode: GimbalCommandMode | None = None
        self._initialized = False
        self._last_t = clock.monotonic_s()
        self._rng = np.random.default_rng(self._cfg.sim_seed)

    def _integrate(self) -> None:
        now = self._clock.monotonic_s()
        dt = now - self._last_t
        self._last_t = now
        if dt <= 0.0:
            return
        if self._mode is GimbalCommandMode.RATE:
            lo = self._cfg.operational_min_deg
            hi = self._cfg.operational_max_deg
        else:
            lo = self._cfg.hardware_min_deg
            hi = self._cfg.hardware_max_deg
        slew = self._cfg.max_hw_slew_rate_deg_per_s
        max_step = slew * dt
        if self._mode is GimbalCommandMode.RATE:
            self._position += min(max(self._velocity * dt, -max_step), max_step)
        elif self._mode is not None:
            tau = self._cfg.sim_time_constant_s
            alpha = 1.0 - math.exp(-dt / tau)
            self._position += min(max((self._target - self._position) * alpha, -max_step), max_step)
        bounded = min(max(self._position, lo), hi)
        if self._mode is GimbalCommandMode.RATE and bounded != self._position:
            self._velocity = 0.0
            self._mode = None
        self._position = bounded

    def initialize(self) -> Result[None, FaultCode]:
        self._initialized = True
        return Ok(None)

    def shutdown(self) -> Result[None, FaultCode]:
        self.stop()
        self._initialized = False
        return Ok(None)

    def find_index(self) -> Result[None, FaultCode]:
        self._integrate()
        self._target = self._cfg.home_deg
        self._position = self._target
        self._mode = GimbalCommandMode.HOME
        return Ok(None)

    def set_position(self, position_deg: float) -> Result[None, FaultCode]:
        self._integrate()
        lo = self._cfg.operational_min_deg
        hi = self._cfg.operational_max_deg
        self._target = min(max(position_deg, lo), hi)
        self._velocity = 0.0
        self._mode = GimbalCommandMode.ABSOLUTE
        return Ok(None)

    def set_velocity(self, velocity_deg_per_s: float) -> Result[None, FaultCode]:
        self._integrate()
        limit = self._cfg.max_hw_slew_rate_deg_per_s
        self._velocity = min(max(velocity_deg_per_s, -limit), limit)
        if (
            self._velocity < 0.0
            and self._position <= self._cfg.operational_min_deg
            or self._velocity > 0.0
            and self._position >= self._cfg.operational_max_deg
        ):
            return self.stop()
        if self._velocity == 0.0:
            return self.stop()
        self._mode = GimbalCommandMode.RATE
        return Ok(None)

    def stop(self) -> Result[None, FaultCode]:
        self._integrate()
        self._velocity = 0.0
        self._mode = None
        return Ok(None)

    def home(self) -> Result[None, FaultCode]:
        self._integrate()
        self._target = self._cfg.home_deg
        self._velocity = 0.0
        self._mode = GimbalCommandMode.HOME
        return Ok(None)

    def stow(self) -> Result[None, FaultCode]:
        self._integrate()
        self._target = self._cfg.stow_deg
        self._velocity = 0.0
        self._mode = GimbalCommandMode.STOW
        return Ok(None)

    def reset_faults(self) -> Result[None, FaultCode]:
        return Ok(None)

    def read_state(self) -> Result[GimbalAxisState, FaultCode]:
        self._integrate()
        noise = float(self._rng.normal(0.0, self._cfg.sim_encoder_noise_deg))
        return Ok(
            GimbalAxisState(
                position_deg=self._position + noise,
                velocity_deg_per_s=self._velocity if self._mode is GimbalCommandMode.RATE else 0.0,
                target_position_deg=self._target,
                sample_timestamp_s=self._last_t,
                controller_timestamp_s=self._last_t,
                motor_on=self._mode is not None,
                closed_loop=True,
                encoder_valid=True,
                at_index=abs(self._position - self._cfg.home_deg) < _STOW_TOLERANCE_DEG,
                position_reached=abs(self._position - self._target) < _STOW_TOLERANCE_DEG,
                scanning=self._mode is GimbalCommandMode.RATE and self._velocity != 0.0,
            )
        )
