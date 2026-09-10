"""Xeryon v1.88 driver for the single elevation axis of an XD-C controller."""

from __future__ import annotations

import importlib
import math
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

from xeryon_vendor.facade import XeryonAxis as _Axis
from xeryon_vendor.facade import XeryonController as _Controller

from flight.hal.interfaces.gimbal import GimbalAxisState
from flight.libs.config import GimbalConfig
from flight.libs.time import Clock
from flight.libs.types import Err, FaultCode, Ok, Result

AxisFactory = Callable[[str, int, str, str], tuple[_Controller, _Axis]]


def _vendor_factory(
    port: str, baud: int, axis_name: str, stage_name: str
) -> tuple[_Controller, _Axis]:
    """Lazily import and configure the pristine vendored module."""
    module = importlib.import_module("xeryon_vendor.Xeryon")
    setattr(module, "DISABLE_WAITING", True)
    setattr(module, "OUTPUT_TO_CONSOLE", False)
    controller: _Controller = module.Xeryon(port, baud)
    axis = controller.addAxis(getattr(module.Stage, stage_name), axis_name)
    return controller, axis


class RealGimbal:
    """Normalized one-axis actuator backed by Xeryon's XD-C Python library."""

    _TIME_MODULUS = 2**16
    _TIME_UNIT_S = 1e-4
    _RATE_QUANTUM_DEG_PER_S = 0.01
    _RATE_DEADBAND_DEG_PER_S = 0.005

    def __init__(
        self,
        clock: Clock,
        cfg: GimbalConfig | None = None,
        *,
        factory: AxisFactory | None = None,
    ) -> None:
        self._clock = clock
        self._cfg = cfg or GimbalConfig()
        self._factory = factory or _vendor_factory
        self._controller: _Controller | None = None
        self._axis: _Axis | None = None
        self._lock = threading.RLock()
        self._initialized = False
        self._shutdown_event = threading.Event()
        self._watchdog: threading.Thread | None = None
        self._target_deg = self._cfg.home_deg
        self._commanded_rate_deg_per_s = 0.0
        self._last_rate_command_s: float | None = None
        self._rate_lease_expired = False
        self._last_epos_counts: float | None = None
        self._last_controller_time: int | None = None
        self._controller_time_unwrapped_ticks: int | None = None
        self._last_feedback_host_s: float | None = None
        self._motor_started_s: float | None = None
        self._duty_fault = False
        self._stow_in_progress = False

    def initialize(self) -> Result[None, FaultCode]:
        """Connect, load deployment settings, configure feedback, and find index."""
        cfg = self._cfg
        settings_path = Path(cfg.settings_default_path)
        if not cfg.serial_port or not cfg.settings_default_path or not settings_path.is_file():
            return Err(FaultCode.GIMBAL_FAULT)
        try:
            controller, axis = self._factory(
                cfg.serial_port, cfg.serial_baud, cfg.axis_name, cfg.stage
            )
            self._controller, self._axis = controller, axis
            controller.start(external_settings_default=str(settings_path))
            units = self._vendor_units_deg()
            if units is not None:
                axis.setUnits(units)
            controller.setMasterSetting("INFO", cfg.feedback_info)
            controller.setMasterSetting("POLI", cfg.feedback_poll_interval_ms)
            axis.setSetting("LLIM", self._position_to_counts(cfg.hardware_min_deg))
            axis.setSetting("HLIM", self._position_to_counts(cfg.hardware_max_deg))
            indexed = axis.findIndex(forceWaiting=True)
            if indexed is False or not all(
                (
                    self._status("isMotorOn"),
                    self._status("isClosedLoop"),
                    self._status("isEncoderValid"),
                )
            ):
                self._safe_controller_stop()
                return Err(FaultCode.GIMBAL_FAULT)
            self._initialized = True
            self._shutdown_event.clear()
            self._watchdog = threading.Thread(
                target=self._watchdog_loop, name="xeryon-gimbal-watchdog", daemon=True
            )
            self._watchdog.start()
            return Ok(None)
        except Exception:  # noqa: BLE001 -- vendor exceptions have no stable base class
            self._safe_controller_stop()
            return Err(FaultCode.GIMBAL_FAULT)

    def shutdown(self) -> Result[None, FaultCode]:
        """Stop all motion and close the Xeryon communication thread."""
        self._shutdown_event.set()
        watchdog = self._watchdog
        if watchdog is not None and watchdog is not threading.current_thread():
            watchdog.join(timeout=1.0)
        with self._lock:
            try:
                if self._controller is not None:
                    self._controller.stopMovements()
                    self._controller.stop()
                self._initialized = False
                return Ok(None)
            except Exception:  # noqa: BLE001 -- normalize every vendor failure at the HAL
                self._initialized = False
                return Err(FaultCode.GIMBAL_FAULT)

    def find_index(self) -> Result[None, FaultCode]:
        """Repeat the blocking index search and validate encoder state."""
        with self._lock:
            if not self._ready():
                return Err(FaultCode.GIMBAL_FAULT)
            try:
                assert self._axis is not None
                found = self._axis.findIndex(forceWaiting=True)
                if found is False or not self._status("isEncoderValid"):
                    return Err(FaultCode.GIMBAL_FAULT)
                return Ok(None)
            except Exception:  # noqa: BLE001 -- normalize every vendor failure at the HAL
                return Err(FaultCode.GIMBAL_FAULT)

    def reset_faults(self) -> Result[None, FaultCode]:
        """Reset the controller and clear locally latched duty/lease faults."""
        with self._lock:
            if self._controller is None:
                return Err(FaultCode.GIMBAL_FAULT)
            try:
                self._controller.reset()
                self._duty_fault = False
                self._rate_lease_expired = False
                self._motor_started_s = None
                return Ok(None)
            except Exception:  # noqa: BLE001 -- normalize every vendor failure at the HAL
                return Err(FaultCode.GIMBAL_FAULT)

    def set_position(self, position_deg: float) -> Result[None, FaultCode]:
        """Queue an operational position clamped to the 0..45 degree imaging range."""
        target = self._clamp(
            position_deg, self._cfg.operational_min_deg, self._cfg.operational_max_deg
        )
        return self._set_position(target, stow=False)

    def set_velocity(self, velocity_deg_per_s: float) -> Result[None, FaultCode]:
        """Command quantized closed-loop scan velocity with safe reversal semantics."""
        with self._lock:
            if not self._ready() or self._motion_forbidden(stow=False):
                return Err(FaultCode.GIMBAL_FAULT)
            rate = self._quantize_rate(
                self._clamp(
                    velocity_deg_per_s,
                    -self._cfg.max_hw_slew_rate_deg_per_s,
                    self._cfg.max_hw_slew_rate_deg_per_s,
                )
            )
            if rate == 0.0:
                return self.stop()
            position = self._cached_position_deg()
            if position is not None and (
                (rate < 0.0 and position <= self._cfg.operational_min_deg)
                or (rate > 0.0 and position >= self._cfg.operational_max_deg)
            ):
                return self.stop()
            try:
                assert self._axis is not None
                previous = self._commanded_rate_deg_per_s
                if previous != 0.0 and (previous > 0.0) != (rate > 0.0):
                    self._axis.stopScan()
                    if not self._wait_until_not_scanning():
                        return Err(FaultCode.GIMBAL_FAULT)
                controller_rate = rate * self._cfg.direction_sign
                self._axis.setSpeed(abs(controller_rate))
                if previous == 0.0 or (previous > 0.0) != (rate > 0.0):
                    self._axis.startScan(1 if controller_rate > 0.0 else -1)
                self._commanded_rate_deg_per_s = rate
                self._last_rate_command_s = self._clock.monotonic_s()
                self._rate_lease_expired = False
                self._stow_in_progress = False
                return Ok(None)
            except Exception:  # noqa: BLE001 -- normalize every vendor failure at the HAL
                return Err(FaultCode.GIMBAL_FAULT)

    def stop(self) -> Result[None, FaultCode]:
        """Stop both scanning and finite position motion."""
        with self._lock:
            if self._controller is None:
                return Err(FaultCode.GIMBAL_FAULT)
            try:
                if self._axis is not None:
                    self._axis.stopScan()
                self._controller.stopMovements()
                self._commanded_rate_deg_per_s = 0.0
                self._last_rate_command_s = None
                self._stow_in_progress = False
                return Ok(None)
            except Exception:  # noqa: BLE001 -- normalize every vendor failure at the HAL
                return Err(FaultCode.GIMBAL_FAULT)

    def home(self) -> Result[None, FaultCode]:
        return self._set_position(self._cfg.home_deg, stow=False)

    def stow(self) -> Result[None, FaultCode]:
        return self._set_position(self._cfg.stow_deg, stow=True)

    def read_state(self) -> Result[GimbalAxisState, FaultCode]:
        """Return one coherent cached EPOS/TIME/status snapshot."""
        with self._lock:
            if not self._ready():
                return Err(FaultCode.GIMBAL_FAULT)
            self._watchdog_tick()
            try:
                raw_epos = float(self._required_data("EPOS"))
                raw_dpos = float(self._required_data("DPOS"))
                raw_time = int(self._required_data("TIME"))
                now = self._clock.monotonic_s()
                ticks = self._time_delta_ticks(raw_time)
                if self._controller_time_unwrapped_ticks is None:
                    self._controller_time_unwrapped_ticks = raw_time
                elif ticks is not None:
                    self._controller_time_unwrapped_ticks += ticks
                velocity: float | None = None
                if self._last_epos_counts is not None and ticks is not None and ticks > 0:
                    delta_deg = self._counts_to_position(raw_epos) - self._counts_to_position(
                        self._last_epos_counts
                    )
                    velocity = delta_deg / (ticks * self._TIME_UNIT_S)
                previous_feedback_s = self._last_feedback_host_s
                time_advanced = self._last_controller_time is None or (
                    ticks is not None and ticks > 0
                )
                self._last_epos_counts = raw_epos
                self._last_controller_time = raw_time
                if time_advanced:
                    self._last_feedback_host_s = now
                stale = (
                    not time_advanced
                    and previous_feedback_s is not None
                    and now - previous_feedback_s > self._cfg.feedback_stale_s
                )
                motor_on = self._status("isMotorOn")
                self._track_duty(motor_on, now)
                thermal_fault = self._status("isThermalProtection1") or self._status(
                    "isThermalProtection2"
                )
                end_limit_fault = (
                    self._status("isAtLeftEnd")
                    or self._status("isAtRightEnd")
                    or self._status("isErrorLimit")
                )
                encoder_valid = self._status("isEncoderValid")
                state = GimbalAxisState(
                    position_deg=self._counts_to_position(raw_epos),
                    velocity_deg_per_s=velocity,
                    target_position_deg=self._counts_to_position(raw_dpos),
                    sample_timestamp_s=now,
                    controller_timestamp_s=(
                        self._controller_time_unwrapped_ticks * self._TIME_UNIT_S
                    ),
                    motor_on=motor_on,
                    closed_loop=self._status("isClosedLoop"),
                    encoder_valid=encoder_valid,
                    at_index=self._status("isEncoderAtIndex"),
                    position_reached=self._status("isPositionReached"),
                    scanning=self._status("isScanning"),
                    thermal_fault=thermal_fault,
                    encoder_fault=self._status("isEncoderError"),
                    end_limit_fault=end_limit_fault,
                    safety_timeout_fault=self._status("isSafetyTimeoutTriggered")
                    or self._duty_fault,
                    position_failure_fault=self._status("isPositionFailTriggered"),
                    feedback_stale=stale,
                    rate_lease_expired=self._rate_lease_expired,
                )
                invalid_status = (
                    not motor_on
                    or not state.closed_loop
                    or not encoder_valid
                    or state.encoder_fault
                    or state.safety_timeout_fault
                    or state.position_failure_fault
                )
                if stale or thermal_fault or end_limit_fault or invalid_status:
                    self._safe_stop_motion()
                    return Err(FaultCode.GIMBAL_FAULT)
                return Ok(state)
            except Exception:  # noqa: BLE001 -- normalize every vendor failure at the HAL
                return Err(FaultCode.GIMBAL_FAULT)

    def watchdog_tick(self) -> None:
        """Run one watchdog check synchronously (test and supervised-loop hook)."""
        with self._lock:
            self._watchdog_tick()

    def _set_position(self, target: float, *, stow: bool) -> Result[None, FaultCode]:
        with self._lock:
            if not self._ready() or self._motion_forbidden(stow=stow):
                return Err(FaultCode.GIMBAL_FAULT)
            try:
                assert self._axis is not None
                self._axis.stopScan()
                bounded = self._clamp(
                    target, self._cfg.hardware_min_deg, self._cfg.hardware_max_deg
                )
                result = self._axis.setDPOS(
                    self._to_controller_degrees(bounded), outputToConsole=False
                )
                if result is False:
                    return Err(FaultCode.GIMBAL_FAULT)
                self._target_deg = bounded
                self._commanded_rate_deg_per_s = 0.0
                self._last_rate_command_s = None
                self._stow_in_progress = stow
                return Ok(None)
            except Exception:  # noqa: BLE001 -- normalize every vendor failure at the HAL
                return Err(FaultCode.GIMBAL_FAULT)

    def _watchdog_loop(self) -> None:
        while not self._shutdown_event.wait(0.05):
            with self._lock:
                self._watchdog_tick()

    def _watchdog_tick(self) -> None:
        if self._axis is None:
            return
        now = self._clock.monotonic_s()
        if (
            self._last_rate_command_s is not None
            and now - self._last_rate_command_s > self._cfg.velocity_lease_s
        ):
            self._rate_lease_expired = True
            self._safe_stop_motion()
        position = self._cached_position_deg()
        if position is not None and (
            (self._commanded_rate_deg_per_s < 0.0 and position <= self._cfg.operational_min_deg)
            or (self._commanded_rate_deg_per_s > 0.0 and position >= self._cfg.operational_max_deg)
        ):
            self._safe_stop_motion()
        self._track_duty(self._status("isMotorOn"), now)

    def _track_duty(self, motor_on: bool, now: float) -> None:
        if not motor_on:
            self._motor_started_s = None
            return
        if self._motor_started_s is None:
            self._motor_started_s = now
        limit = (
            self._cfg.motor_duty_limit_s
            if self._stow_in_progress
            else self._cfg.motor_duty_limit_s - self._cfg.motor_duty_stow_reserve_s
        )
        if now - self._motor_started_s >= limit:
            self._duty_fault = True
            self._safe_stop_motion()

    def _motion_forbidden(self, *, stow: bool) -> bool:
        if self._duty_fault and not stow:
            return True
        if self._motor_started_s is None:
            return False
        elapsed = self._clock.monotonic_s() - self._motor_started_s
        if stow:
            return elapsed >= self._cfg.motor_duty_limit_s
        return elapsed >= (self._cfg.motor_duty_limit_s - self._cfg.motor_duty_stow_reserve_s)

    def _safe_stop_motion(self) -> None:
        try:
            if self._controller is not None:
                self._controller.stopMovements()
        except Exception:  # noqa: BLE001 -- best-effort safety stop
            pass
        self._commanded_rate_deg_per_s = 0.0
        self._last_rate_command_s = None

    def _safe_controller_stop(self) -> None:
        self._safe_stop_motion()
        try:
            if self._controller is not None:
                self._controller.stop()
        except Exception:  # noqa: BLE001 -- best-effort controller teardown
            pass

    def _wait_until_not_scanning(self) -> bool:
        deadline = time.monotonic() + self._cfg.feedback_stale_s
        while self._status("isScanning"):
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.005)
        return True

    def _required_data(self, tag: str) -> float | int | str:
        assert self._axis is not None
        value = self._axis.getData(tag)
        if value is None:
            raise ValueError(f"Xeryon feedback {tag} is unavailable")
        return value

    def _cached_position_deg(self) -> float | None:
        if self._axis is None:
            return None
        try:
            value = self._axis.getData("EPOS")
            return None if value is None else self._counts_to_position(float(value))
        except Exception:  # noqa: BLE001 -- cached vendor data may fail arbitrarily
            return None

    def _status(self, method: str) -> bool:
        if self._axis is None:
            return False
        try:
            function = getattr(self._axis, method, None)
            return bool(function()) if function is not None else False
        except Exception:  # noqa: BLE001 -- invalid vendor status must fail safe
            self._safe_stop_motion()
            return False

    def _time_delta_ticks(self, current: int) -> int | None:
        if self._last_controller_time is None:
            return None
        return (current - self._last_controller_time) % self._TIME_MODULUS

    def _to_controller_degrees(self, position_deg: float) -> float:
        return self._cfg.direction_sign * (position_deg - self._cfg.index_offset_deg)

    def _position_to_counts(self, position_deg: float) -> int:
        controller_deg = self._to_controller_degrees(position_deg)
        return round(controller_deg * self._cfg.encoder_counts_per_rev / 360.0)

    def _counts_to_position(self, counts: float) -> float:
        controller_deg = counts * 360.0 / self._cfg.encoder_counts_per_rev
        return controller_deg * self._cfg.direction_sign + self._cfg.index_offset_deg

    def _quantize_rate(self, rate: float) -> float:
        if abs(rate) < self._RATE_DEADBAND_DEG_PER_S:
            return 0.0
        quanta = math.floor(abs(rate) / self._RATE_QUANTUM_DEG_PER_S + 0.5)
        return math.copysign(quanta * self._RATE_QUANTUM_DEG_PER_S, rate)

    def _vendor_units_deg(self) -> object | None:
        if self._factory is not _vendor_factory:
            return None
        try:
            return cast(object, importlib.import_module("xeryon_vendor.Xeryon").Units.deg)
        except AttributeError, ImportError:
            return None

    def _ready(self) -> bool:
        return self._initialized and self._axis is not None and self._controller is not None

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return min(max(float(value), low), high)
