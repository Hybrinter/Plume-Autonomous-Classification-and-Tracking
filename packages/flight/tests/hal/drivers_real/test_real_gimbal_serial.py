"""RealGimbal rate-command and fail-closed scaffolding tests."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from flight.hal.drivers_real import RealGimbal
from flight.hal.drivers_real.gimbal import DutyCreditBucket, VendorFactory
from flight.hal.interfaces import GimbalRateCommand
from flight.libs.config import GimbalConfig, XeryonConfig
from flight.libs.time import ManualClock
from flight.libs.types import Err, FaultCode, Ok


@dataclass
class _FakeGate:
    inhibited: bool = False
    requests: list[str] | None = None

    def __post_init__(self) -> None:
        self.requests = []

    def request_inhibit(self, reason: str) -> Ok[None]:
        assert self.requests is not None
        self.requests.append(reason)
        self.inhibited = True
        return Ok(None)

    def inhibit_confirmed(self) -> Ok[bool]:
        return Ok(self.inhibited)


class _FakeAxis:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.motor_on = False
        self.closed_loop = True
        self.encoder_valid = True
        self.data: dict[str, object] = {"EPOS": 43_200, "TIME": 42, "STAT": 0x1234}

    def setSetting(self, tag: str, value: str) -> None:  # noqa: N802
        self.calls.append(("setSetting", (tag, value)))

    def startScan(self, direction: int) -> None:  # noqa: N802
        self.calls.append(("startScan", direction))
        self.motor_on = True

    def stopScan(self) -> None:  # noqa: N802
        self.calls.append(("stopScan", None))
        self.motor_on = False

    def getData(self, tag: str) -> object:  # noqa: N802
        return self.data.get(tag)

    def isMotorOn(self) -> bool:  # noqa: N802
        return self.motor_on

    def isClosedLoop(self) -> bool:  # noqa: N802
        return self.closed_loop

    def isEncoderValid(self) -> bool:  # noqa: N802
        return self.encoder_valid

    def isEncoderError(self) -> bool:  # noqa: N802
        return False

    def isThermalProtection1(self) -> bool:  # noqa: N802
        return False

    def isThermalProtection2(self) -> bool:  # noqa: N802
        return False

    def isSafetyTimeoutTriggered(self) -> bool:  # noqa: N802
        return False


class _FakeCommunication:
    def __init__(self) -> None:
        self.closed = False

    def closeCommunication(self) -> None:  # noqa: N802
        self.closed = True


class _FakeController:
    def __init__(self, axis: _FakeAxis) -> None:
        self.axis = axis
        self.communication = _FakeCommunication()
        self.stop_called = False

    def getCommunication(self) -> _FakeCommunication:  # noqa: N802
        return self.communication

    def stop(self) -> None:
        self.stop_called = True


def _enabled_cfg(settings_file_path: str = __file__) -> GimbalConfig:
    """Build a fully qualified fake-hardware config."""
    return GimbalConfig(
        xeryon=XeryonConfig(
            serial_port="FAKE",
            settings_file_path=settings_file_path,
            motion_enabled=True,
            vendor_license_audited=True,
            python314_audited=True,
            watchdog_validated=True,
            stow_bench_validated=True,
        )
    )


def test_set_torque_refuses_unimplemented_amp() -> None:
    """The real stub must not claim motor containment or drive authority."""
    gimbal = RealGimbal(clock=ManualClock(), cfg=GimbalConfig())
    result = gimbal.set_torque(0.5, valid_until_s=1.0)
    assert isinstance(result, Err)
    assert result.error is FaultCode.GIMBAL_FAULT


def test_motion_enablement_requires_external_settings_file() -> None:
    """A qualified flag set cannot start the controller without audited settings."""
    missing = str(Path(__file__).with_name("missing_xeryon_settings.txt"))
    axis = _FakeAxis()
    controller = _FakeController(axis)
    gate = _FakeGate()
    gimbal = RealGimbal(
        clock=ManualClock(),
        cfg=_enabled_cfg(missing),
        watchdog_gate=gate,
        vendor_factory=cast(VendorFactory, lambda _cfg: (controller, axis)),
    )
    result = gimbal.set_rate(GimbalRateCommand(1.0, 1.0))
    assert isinstance(result, Err)
    assert result.error is FaultCode.GIMBAL_FAULT


def test_goto_angle_latches_target_not_encoder() -> None:
    """goto_angle stores a target without fabricating disconnected feedback."""
    gimbal = RealGimbal(clock=ManualClock(), cfg=GimbalConfig())
    assert isinstance(gimbal.goto_angle(500.0), Ok)
    pos = gimbal.read_position()
    assert isinstance(pos, Err)
    assert pos.error is FaultCode.GIMBAL_ENCODER_INVALID
    assert gimbal._target_el_deg == 90.0


def test_stow_does_not_teleport_encoder() -> None:
    """stow() latches its target without inventing an encoder sample."""
    gimbal = RealGimbal(clock=ManualClock(), cfg=GimbalConfig())
    assert isinstance(gimbal.stow(), Ok)
    pos = gimbal.read_position()
    assert isinstance(pos, Err)
    assert pos.error is FaultCode.GIMBAL_ENCODER_INVALID
    switch = gimbal.read_stow_switch()
    assert isinstance(switch, Ok)
    assert switch.value is False


def test_rate_quantization_uses_nearest_step_and_half_step_deadband() -> None:
    """The 0.01 deg/s command quantum has a symmetric deadband."""
    assert RealGimbal.quantize_rate(0.004, 0.01) == 0.0
    assert RealGimbal.quantize_rate(0.005, 0.01) == 0.0
    assert RealGimbal.quantize_rate(0.006, 0.01) == 0.01
    assert RealGimbal.quantize_rate(-0.006, 0.01) == -0.01


def test_duty_credit_locks_until_full_recovery() -> None:
    """HV credit drains on motor-on and unlocks only at full recovery."""
    bucket = DutyCreditBucket().advance(0.0, False).advance(120.0, True)
    assert bucket.locked_out is True
    assert bucket.credit_s == 0.0
    recovered = bucket.advance(239.0, False)
    assert recovered.locked_out is True
    assert recovered.credit_s == 119.0
    unlocked = recovered.advance(240.0, False)
    assert unlocked.locked_out is False
    assert unlocked.credit_s == 120.0


def test_invalid_encoder_stops_and_requests_external_inhibit() -> None:
    """Invalid feedback prevents scan and asks the independent gate to inhibit."""
    axis = _FakeAxis()
    axis.encoder_valid = False
    controller = _FakeController(axis)
    gate = _FakeGate()
    gimbal = RealGimbal(
        clock=ManualClock(),
        cfg=_enabled_cfg(),
        watchdog_gate=gate,
        vendor_factory=cast(VendorFactory, lambda _cfg: (controller, axis)),
    )
    result = gimbal.set_rate(GimbalRateCommand(1.0, 1.0))
    assert isinstance(result, Err)
    assert result.error is FaultCode.GIMBAL_ENCODER_INVALID
    assert ("startScan", 1) not in axis.calls
    assert gate.requests == ["gimbal local fault stop"]


def test_read_position_rejects_closed_loop_loss_while_motor_on() -> None:
    """Position feedback cannot be accepted while an energized loop is open."""
    axis = _FakeAxis()
    controller = _FakeController(axis)
    gate = _FakeGate()
    gimbal = RealGimbal(
        clock=ManualClock(),
        cfg=_enabled_cfg(),
        watchdog_gate=gate,
        vendor_factory=cast(VendorFactory, lambda _cfg: (controller, axis)),
        time_mapper=lambda raw: (float(raw), 0.0),
    )
    assert isinstance(gimbal.set_rate(GimbalRateCommand(1.0, 1.0)), Ok)
    axis.closed_loop = False
    position = gimbal.read_position()
    assert isinstance(position, Err)
    assert position.error is FaultCode.GIMBAL_CLOSED_LOOP_LOSS
    assert gate.requests == ["gimbal local fault stop"]


def test_fake_vendor_rate_sequence_feedback_mapping_and_safe_shutdown() -> None:
    """Rate commands use stop/set-speed/scan and shutdown never calls vendor stop."""
    axis = _FakeAxis()
    controller = _FakeController(axis)
    gate = _FakeGate()
    gimbal = RealGimbal(
        clock=ManualClock(),
        cfg=_enabled_cfg(),
        watchdog_gate=gate,
        vendor_factory=cast(VendorFactory, lambda _cfg: (controller, axis)),
        time_mapper=lambda raw: (float(raw) * 0.01, 1.0e-5),
    )
    assert isinstance(gimbal.set_rate(GimbalRateCommand(1.234, 1.0)), Ok)
    assert axis.calls[-2:] == [("setSetting", ("SSPD", "123")), ("startScan", 1)]
    assert isinstance(gimbal.set_rate(GimbalRateCommand(-1.23, 2.0)), Ok)
    assert axis.calls[-3:] == [
        ("stopScan", None),
        ("setSetting", ("SSPD", "123")),
        ("startScan", -1),
    ]
    position = gimbal.read_position()
    assert isinstance(position, Ok)
    assert position.value.el_deg == 180.0
    assert position.value.raw_controller_time_s == 42.0
    assert position.value.time_mapping_uncertainty_s == 1.0e-5
    health = gimbal.read_health()
    assert isinstance(health, Ok)
    assert health.value.requested_rate_deg_per_s == -1.23
    assert health.value.quantized_rate_deg_per_s == -1.23
    assert isinstance(gimbal.shutdown(), Ok)
    assert controller.stop_called is False
    assert controller.communication.closed is True
    assert gate.requests == ["shutdown"]
