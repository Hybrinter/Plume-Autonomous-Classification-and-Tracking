"""Fail-closed Xeryon/XD-C gimbal adapter.

The vendored Xeryon v1.88 module is imported with this driver. Serial I/O still
waits until a rate command connects with audited production prerequisites. The
adapter uses ``setSpeed`` plus ``startScan`` for motion, and ``stopScan`` then
``stopMovements`` to halt; it never calls the vendor library's ``stop()`` helper
because that helper also performs controller shutdown actions that may home the
stage.

The legacy torque methods remain an explicit rejected compatibility surface
for detailed-plant SIL. Production hardware is driven through ``set_rate``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, cast

from xeryon_vendor import Xeryon

from flight.hal.interfaces.gimbal import (
    ExternalWatchdogGate,
    GimbalHealth,
    GimbalPosition,
    GimbalRateCommand,
)
from flight.libs.config import GimbalConfig, XeryonConfig
from flight.libs.time import Clock
from flight.libs.types import Err, FaultCode, Ok, Result

_VENDOR_ERRORS = (AttributeError, OSError, RuntimeError, TypeError, ValueError)


class _VendorAxis(Protocol):
    """Small subset of the vendor axis API used by the adapter."""

    def setUnits(self, units: object) -> object: ...  # noqa: N802

    def setSetting(self, tag: str, value: str) -> object: ...  # noqa: N802

    def setSpeed(self, speed: float) -> object: ...  # noqa: N802

    def startScan(self, direction: int) -> object: ...  # noqa: N802

    def stopScan(self) -> object: ...  # noqa: N802

    def getData(self, tag: str) -> object: ...  # noqa: N802

    def isMotorOn(self) -> bool: ...  # noqa: N802

    def isClosedLoop(self) -> bool: ...  # noqa: N802

    def isEncoderValid(self) -> bool: ...  # noqa: N802

    def isEncoderError(self) -> bool: ...  # noqa: N802

    def isThermalProtection1(self) -> bool: ...  # noqa: N802

    def isThermalProtection2(self) -> bool: ...  # noqa: N802

    def isSafetyTimeoutTriggered(self) -> bool: ...  # noqa: N802


class _VendorCommunication(Protocol):
    def closeCommunication(self) -> object: ...  # noqa: N802


class _VendorController(Protocol):
    def addAxis(self, stage: object, axis_letter: str) -> _VendorAxis: ...  # noqa: N802

    def start(
        self,
        external_communication_thread: bool = False,
        external_settings_default: str | None = None,
    ) -> object: ...

    def stopMovements(self) -> object: ...  # noqa: N802

    def getCommunication(self) -> _VendorCommunication: ...  # noqa: N802


VendorFactory = Callable[[XeryonConfig], tuple[_VendorController, _VendorAxis]]
TimeMapper = Callable[[float], tuple[float, float]]


@dataclass(frozen=True, slots=True)
class DutyCreditBucket:
    """Conservative HV on-time credit bucket.

    Credit drains one second per second while the controller reports motor-on
    and replenishes at one second per second while motor-off. Once exhausted,
    the bucket remains locked until the full credit amount has recovered.
    """

    max_credit_s: float = 120.0
    credit_s: float = 120.0
    last_update_s: float | None = None
    locked_out: bool = False

    def advance(self, now_s: float, motor_on: bool) -> DutyCreditBucket:
        """Advance credit using monotonic time; backwards time is fail-closed."""
        if not math.isfinite(now_s):
            return replace(self, locked_out=True, credit_s=0.0)
        if self.last_update_s is None:
            return replace(self, last_update_s=now_s)
        dt_s = now_s - self.last_update_s
        if dt_s < 0.0:
            return replace(self, last_update_s=now_s, locked_out=True, credit_s=0.0)
        if self.locked_out:
            credit = min(self.max_credit_s, self.credit_s + dt_s)
            return replace(
                self,
                credit_s=credit,
                last_update_s=now_s,
                locked_out=credit < self.max_credit_s,
            )
        if motor_on:
            credit = max(0.0, self.credit_s - dt_s)
            return replace(
                self,
                credit_s=credit,
                last_update_s=now_s,
                locked_out=credit <= 0.0,
            )
        return replace(
            self,
            credit_s=min(self.max_credit_s, self.credit_s + dt_s),
            last_update_s=now_s,
        )

    @property
    def motion_allowed(self) -> bool:
        """True only while credit remains and the full-recovery lock is clear."""
        return self.credit_s > 0.0 and not self.locked_out


class RealGimbal:
    """Xeryon rate-command driver with fail-closed startup and shutdown."""

    def __init__(
        self,
        clock: Clock,
        cfg: GimbalConfig | None = None,
        *,
        watchdog_gate: ExternalWatchdogGate | None = None,
        vendor_factory: VendorFactory | None = None,
        time_mapper: TimeMapper | None = None,
        stow_switch_reader: Callable[[], bool] | None = None,
    ) -> None:
        """Construct without opening a serial port."""
        self._cfg = cfg if cfg is not None else GimbalConfig()
        self._xcfg = self._cfg.xeryon
        self._clock = clock
        self._watchdog_gate = watchdog_gate
        self._vendor_factory = vendor_factory
        self._time_mapper = time_mapper
        self._stow_switch_reader = stow_switch_reader
        self._controller: _VendorController | None = None
        self._axis: _VendorAxis | None = None
        self._el_deg = 0.0
        self._target_el_deg = 0.0
        self._stow_commanded = False
        self._stow_started_s: float | None = None
        self._last_direction = 0
        self._last_speed_requested: float | None = None
        self._last_speed_quantized = 0.0
        self._command_valid_until_s: float | None = None
        self._inhibited = True
        self._last_feedback_s: float | None = None
        self._last_status_bits = 0
        self._sequence = 0
        self._duty = DutyCreditBucket(max_credit_s=self._xcfg.hv_max_on_s)

    @property
    def duty_credit(self) -> DutyCreditBucket:
        """Current immutable duty-credit state for telemetry/tests."""
        return self._duty

    @staticmethod
    def quantize_rate(rate_deg_per_s: float, quantum_deg_per_s: float) -> float:
        """Round to nearest quantum with a half-step deadband around zero."""
        if not math.isfinite(rate_deg_per_s) or not math.isfinite(quantum_deg_per_s):
            return math.nan
        if quantum_deg_per_s <= 0.0 or abs(rate_deg_per_s) <= quantum_deg_per_s / 2.0:
            return 0.0
        magnitude = math.floor(abs(rate_deg_per_s) / quantum_deg_per_s + 0.5)
        return math.copysign(magnitude * quantum_deg_per_s, rate_deg_per_s)

    def _position_to_counts(self, position_deg: float) -> int:
        """Convert signed degrees to controller encoder counts."""
        return int(round(position_deg / 360.0 * self._xcfg.controller_counts_per_rev))

    def _connect(self) -> Result[None, FaultCode]:
        """Lazily construct the vendor controller and configure polling."""
        if self._axis is not None:
            return Ok(None)
        if not self._xcfg.serial_port:
            return Err(FaultCode.GIMBAL_FAULT)
        if not self._xcfg.settings_file_path or not Path(self._xcfg.settings_file_path).is_file():
            return Err(FaultCode.GIMBAL_FAULT)
        try:
            if self._vendor_factory is not None:
                self._controller, self._axis = self._vendor_factory(self._xcfg)
            else:
                Xeryon.DISABLE_WAITING = True
                Xeryon.OUTPUT_TO_CONSOLE = False
                controller = cast(
                    _VendorController,
                    Xeryon.Xeryon(self._xcfg.serial_port, self._xcfg.baudrate),
                )
                stage = getattr(Xeryon.Stage, self._xcfg.vendor_stage)
                self._controller = controller
                self._axis = controller.addAxis(stage, self._xcfg.axis_letter)
                self._controller.start(
                    external_settings_default=self._xcfg.settings_file_path,
                )
            assert self._axis is not None
            # INFO/POLI are initial settings; measured cadence must still be
            # qualified before these values are treated as final.
            self._axis.setUnits(Xeryon.Units.deg)
            self._axis.setSetting("INFO", str(self._xcfg.feedback_info_level))
            self._axis.setSetting("POLI", str(int(self._xcfg.feedback_poll_interval_ms)))
            lo = self._position_to_counts(self._cfg.el_hw_min_deg)
            hi = self._position_to_counts(self._cfg.el_hw_max_deg)
            self._axis.setSetting("LLIM", str(min(lo, hi)))
            self._axis.setSetting("HLIM", str(max(lo, hi)))
            return Ok(None)
        except AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError:
            self._controller = None
            self._axis = None
            return Err(FaultCode.GIMBAL_CONTROLLER_ERROR)

    def _gate_confirmed(self) -> Result[bool, FaultCode]:
        """Read independent physical inhibit evidence."""
        if self._watchdog_gate is None:
            return Err(FaultCode.GIMBAL_WATCHDOG_UNCONFIRMED)
        try:
            return self._watchdog_gate.inhibit_confirmed()
        except OSError, RuntimeError:
            return Err(FaultCode.GIMBAL_WATCHDOG_UNCONFIRMED)

    def _axis_status(self) -> Result[tuple[int, bool, bool, bool], FaultCode]:
        """Read raw status and essential safety bits from the vendor axis."""
        axis = self._axis
        if axis is None:
            return Err(FaultCode.GIMBAL_CONTROLLER_ERROR)
        try:
            raw = axis.getData("STAT")
            bits = int(str(raw)) if raw is not None else 0
            return Ok(
                (
                    bits,
                    bool(axis.isMotorOn()),
                    bool(axis.isClosedLoop()),
                    bool(axis.isEncoderValid()),
                )
            )
        except OSError, RuntimeError, TypeError, ValueError:
            return Err(FaultCode.GIMBAL_CONTROLLER_ERROR)

    def _refresh_duty(self, motor_on: bool) -> None:
        """Update duty credit from status without granting credit on errors."""
        self._duty = self._duty.advance(self._clock.monotonic_s(), motor_on)

    def _halt_motion(self) -> None:
        """Stop scan velocity and finite moves without vendor ``stop()``."""
        axis = self._axis
        if axis is not None:
            try:
                axis.stopScan()
            except _VENDOR_ERRORS:
                pass
        controller = self._controller
        if controller is not None:
            try:
                controller.stopMovements()
            except _VENDOR_ERRORS:
                pass

    def _stop_local(self) -> None:
        """Issue non-homing stop and request independent drive inhibition."""
        self._halt_motion()
        self._last_direction = 0
        self._last_speed_quantized = 0.0
        self._inhibited = True
        gate = self._watchdog_gate
        if gate is not None:
            try:
                gate.request_inhibit("gimbal local fault stop")
            except OSError, RuntimeError:
                pass

    def set_torque(
        self, tau_nm: float, valid_until_s: float | None = None
    ) -> Result[None, FaultCode]:
        """Reject torque: production hardware is rate-command only."""
        del tau_nm, valid_until_s
        return Err(FaultCode.GIMBAL_FAULT)

    def set_rate(self, command: GimbalRateCommand) -> Result[None, FaultCode]:
        """Quantize and send a leased signed rate command."""
        now = self._clock.monotonic_s()
        if not self._xcfg.prerequisites_ready:
            return Err(FaultCode.GIMBAL_FAULT)
        if not math.isfinite(command.rate_deg_per_s) or not math.isfinite(command.valid_until_s):
            return Err(FaultCode.GIMBAL_CONTROLLER_ERROR)
        if command.valid_until_s <= now:
            return Err(FaultCode.GIMBAL_SAFETY_TIMEOUT)
        if abs(command.rate_deg_per_s) > self._xcfg.software_tracking_limit_deg_per_s + 1e-12:
            return Err(FaultCode.GIMBAL_FAULT)
        gate = self._gate_confirmed()
        if isinstance(gate, Err):
            return gate
        if gate.value:
            return Err(FaultCode.GIMBAL_WATCHDOG_UNCONFIRMED)
        connected = self._connect()
        if isinstance(connected, Err):
            return connected
        status = self._axis_status()
        if isinstance(status, Err):
            self._stop_local()
            return status
        _bits, motor_on, closed_loop, encoder_valid = status.value
        self._refresh_duty(motor_on)
        if motor_on and not closed_loop:
            self._stop_local()
            return Err(FaultCode.GIMBAL_CLOSED_LOOP_LOSS)
        if not encoder_valid:
            self._stop_local()
            return Err(FaultCode.GIMBAL_ENCODER_INVALID)
        if not self._duty.motion_allowed:
            self._stop_local()
            return Err(FaultCode.GIMBAL_DUTY_EXHAUSTED)
        axis = self._axis
        assert axis is not None
        quantized = self.quantize_rate(command.rate_deg_per_s, self._xcfg.command_quantum_deg_per_s)
        if not math.isfinite(quantized):
            return Err(FaultCode.GIMBAL_CONTROLLER_ERROR)
        self._last_speed_requested = command.rate_deg_per_s
        direction = 0 if quantized == 0.0 else (1 if quantized > 0.0 else -1)
        try:
            if direction == 0:
                self._halt_motion()
                self._last_direction = 0
                self._last_speed_quantized = 0.0
                self._inhibited = True
            else:
                if self._last_direction not in (0, direction):
                    axis.stopScan()
                    if axis.isMotorOn():
                        self._stop_local()
                        return Err(FaultCode.GIMBAL_CONTROLLER_ERROR)
                if self._last_direction != direction or self._last_speed_quantized != quantized:
                    axis.setSpeed(abs(quantized))
                if self._last_direction != direction:
                    axis.startScan(direction)
                self._last_direction = direction
                self._last_speed_quantized = quantized
                self._inhibited = False
            self._command_valid_until_s = command.valid_until_s
            return Ok(None)
        except OSError, RuntimeError, TypeError, ValueError:
            self._stop_local()
            return Err(FaultCode.GIMBAL_CONTROLLER_ERROR)

    def inhibit(self, reason: str) -> Result[GimbalHealth, FaultCode]:
        """Stop scanning and require independent watchdog confirmation."""
        try:
            self._halt_motion()
            if self._watchdog_gate is None:
                return Err(FaultCode.GIMBAL_WATCHDOG_UNCONFIRMED)
            requested = self._watchdog_gate.request_inhibit(reason)
            if isinstance(requested, Err):
                return Err(FaultCode.GIMBAL_WATCHDOG_UNCONFIRMED)
            confirmed = self._watchdog_gate.inhibit_confirmed()
            if isinstance(confirmed, Err) or not confirmed.value:
                return Err(FaultCode.GIMBAL_WATCHDOG_UNCONFIRMED)
        except OSError, RuntimeError, TypeError:
            return Err(FaultCode.GIMBAL_WATCHDOG_UNCONFIRMED)
        now = self._clock.monotonic_s()
        self._command_valid_until_s = now
        self._last_direction = 0
        self._last_speed_quantized = 0.0
        self._inhibited = True
        return Ok(self._health(watchdog_confirmed=True))

    def read_health(self) -> Result[GimbalHealth, FaultCode]:
        """Return controller status, lease, duty, and watchdog evidence."""
        axis = self._axis
        if axis is None:
            return Ok(self._health(watchdog_confirmed=False))
        status = self._axis_status()
        if isinstance(status, Err):
            self._stop_local()
            return status
        bits, motor_on, closed_loop, encoder_valid = status.value
        self._refresh_duty(motor_on)
        if self._duty.locked_out:
            self._stop_local()
            return Err(FaultCode.GIMBAL_DUTY_EXHAUSTED)
        if motor_on and not closed_loop:
            self._stop_local()
            return Err(FaultCode.GIMBAL_CLOSED_LOOP_LOSS)
        if not encoder_valid:
            self._stop_local()
            return Err(FaultCode.GIMBAL_ENCODER_INVALID)
        if (
            self._command_valid_until_s is not None
            and self._clock.monotonic_s() > self._command_valid_until_s
            and not self._inhibited
        ):
            self._stop_local()
            return Err(FaultCode.GIMBAL_SAFETY_TIMEOUT)
        gate = self._gate_confirmed()
        gate_ok = isinstance(gate, Ok) and gate.value
        try:
            thermal = bool(axis.isThermalProtection1() or axis.isThermalProtection2())
            timeout = bool(axis.isSafetyTimeoutTriggered())
        except OSError, RuntimeError:
            self._stop_local()
            return Err(FaultCode.GIMBAL_CONTROLLER_ERROR)
        if thermal:
            self._stop_local()
            return Err(FaultCode.GIMBAL_THERMAL)
        if timeout:
            self._stop_local()
            return Err(FaultCode.GIMBAL_SAFETY_TIMEOUT)
        return Ok(
            self._health(
                feedback_valid=encoder_valid,
                status_bits=bits,
                motor_on=motor_on,
                closed_loop=closed_loop,
                watchdog_confirmed=gate_ok,
            )
        )

    def _health(
        self,
        *,
        feedback_valid: bool = False,
        status_bits: int | None = None,
        motor_on: bool = False,
        closed_loop: bool = False,
        watchdog_confirmed: bool = False,
    ) -> GimbalHealth:
        """Build a compact health record without vendor calls."""
        return GimbalHealth(
            feedback_valid=feedback_valid,
            last_feedback_s=self._last_feedback_s,
            command_valid_until_s=self._command_valid_until_s,
            inhibited=self._inhibited,
            inhibit_confirmed=watchdog_confirmed,
            controller_status_bits=self._last_status_bits if status_bits is None else status_bits,
            motor_on=motor_on,
            closed_loop=closed_loop,
            duty_credit_s=self._duty.credit_s,
            duty_locked_out=self._duty.locked_out,
            watchdog_gate_confirmed=watchdog_confirmed,
            time_mapping_valid=self._time_mapper is not None,
            requested_rate_deg_per_s=self._last_speed_requested,
            quantized_rate_deg_per_s=self._last_speed_quantized,
        )

    def goto_angle(self, el_deg: float) -> Result[None, FaultCode]:
        """Latch a travel-clamped target; no vendor homing/index side effect."""
        cfg = self._cfg
        self._target_el_deg = min(max(el_deg, cfg.el_hw_min_deg), cfg.el_hw_max_deg)
        self._stow_commanded = False
        return Ok(None)

    def home(self) -> Result[None, FaultCode]:
        """Latch the configured home target without commanding vendor HOME."""
        return self.goto_angle(self._cfg.home_el_deg)

    def stow(self) -> Result[None, FaultCode]:
        """Start bounded stow bookkeeping; motion uses ``stow_reference_step``."""
        result = self.goto_angle(self._cfg.stow_el_deg)
        self._stow_commanded = True
        self._stow_started_s = self._clock.monotonic_s()
        return result

    def stow_reference_step(self, now_s: float | None = None) -> Result[bool, FaultCode]:
        """Advance constrained stow without an index search."""
        now = self._clock.monotonic_s() if now_s is None else now_s
        if not self._stow_commanded:
            return Err(FaultCode.GIMBAL_FAULT)
        switch = self.read_stow_switch()
        if isinstance(switch, Err):
            return switch
        if switch.value:
            inhibited = self.inhibit("stow reached")
            if isinstance(inhibited, Err):
                return inhibited
            return Ok(True)
        started = self._stow_started_s
        if started is None or now - started > self._xcfg.stow_timeout_s:
            self.inhibit("bounded stow timeout")
            return Err(FaultCode.GIMBAL_SAFETY_TIMEOUT)
        position = self.read_position()
        if isinstance(position, Err):
            self.inhibit("stow feedback unavailable")
            return position
        direction = -1.0 if position.value.el_deg > self._cfg.stow_el_deg else 1.0
        command_result = self.set_rate(
            GimbalRateCommand(
                rate_deg_per_s=direction * self._xcfg.stow_reference_rate_deg_per_s,
                valid_until_s=min(
                    now + self._xcfg.feedback_max_age_s,
                    started + self._xcfg.stow_timeout_s,
                ),
            )
        )
        if isinstance(command_result, Err):
            return command_result
        return Ok(False)

    def read_position(self) -> Result[GimbalPosition, FaultCode]:
        """Read a mapped controller sample; reject unmappable vendor timestamps."""
        axis = self._axis
        if axis is None:
            return Err(FaultCode.GIMBAL_ENCODER_INVALID)
        health = self.read_health()
        if isinstance(health, Err):
            return health
        try:
            raw_position = axis.getData("EPOS")
            raw_time = axis.getData("TIME")
            status = axis.getData("STAT")
            if raw_position is None or raw_time is None:
                self._stop_local()
                return Err(FaultCode.GIMBAL_ENCODER_INVALID)
            if self._time_mapper is None:
                self._stop_local()
                return Err(FaultCode.GIMBAL_TIME_MAPPING)
            controller_time = float(str(raw_time))
            mapped_time, uncertainty = self._time_mapper(controller_time)
            if not math.isfinite(mapped_time) or not math.isfinite(uncertainty):
                self._stop_local()
                return Err(FaultCode.GIMBAL_TIME_MAPPING)
            if uncertainty > self._xcfg.timing_uncertainty_max_s:
                self._stop_local()
                return Err(FaultCode.GIMBAL_TIME_MAPPING)
            self._last_status_bits = int(str(status)) if status is not None else 0
            self._sequence += 1
            self._last_feedback_s = mapped_time
            return Ok(
                GimbalPosition(
                    el_deg=float(str(raw_position)) * 360.0 / self._xcfg.controller_counts_per_rev,
                    timestamp_s=mapped_time,
                    raw_controller_time_s=controller_time,
                    sequence=self._sequence,
                    status_bits=self._last_status_bits,
                    time_mapping_uncertainty_s=uncertainty,
                )
            )
        except OSError, RuntimeError, TypeError, ValueError:
            self._stop_local()
            return Err(FaultCode.GIMBAL_ENCODER_INVALID)

    def read_stow_switch(self) -> Result[bool, FaultCode]:
        """Read constrained stow state without commanding an index search."""
        if self._stow_switch_reader is not None:
            try:
                return Ok(bool(self._stow_switch_reader()))
            except OSError, RuntimeError:
                return Err(FaultCode.GIMBAL_FAULT)
        return Ok(self._stow_commanded and abs(self._el_deg - self._cfg.stow_el_deg) < 0.5)

    def shutdown(self) -> Result[None, FaultCode]:
        """Confirm external inhibit, then close transport without vendor ``stop``."""
        inhibited = self.inhibit("shutdown")
        if isinstance(inhibited, Err):
            return inhibited
        controller = self._controller
        if controller is not None:
            try:
                controller.getCommunication().closeCommunication()
            except OSError, RuntimeError, TypeError:
                return Err(FaultCode.GIMBAL_CONTROLLER_ERROR)
        return Ok(None)
