# ruff: noqa: N802, N803

"""RealGimbal behavior tests against a fake Xeryon v1.88 controller."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from flight.hal.drivers_real.gimbal import RealGimbal
from flight.libs.config import GimbalConfig
from flight.libs.time import ManualClock
from flight.libs.types import Err, FaultCode, Ok


class FakeAxis:
    def __init__(self) -> None:
        self.data: dict[str, float | int | str | None] = {"EPOS": 0, "DPOS": 0, "TIME": 0}
        self.flags = {
            "motor": True,
            "closed": True,
            "index": True,
            "encoder": True,
            "reached": True,
            "scanning": False,
            "thermal1": False,
            "thermal2": False,
            "encoder_error": False,
            "left": False,
            "right": False,
            "error_limit": False,
            "safety": False,
            "position_fail": False,
        }
        self.calls: list[tuple[Any, ...]] = []
        self.index_result: bool | None = True

    def setUnits(self, units: object) -> None:  # noqa: N802
        self.calls.append(("setUnits", units))

    def setSetting(self, tag: str, value: object) -> None:  # noqa: N802
        self.calls.append(("setSetting", tag, value))

    def findIndex(self, forceWaiting: bool = False, direction: int = 0) -> bool | None:  # noqa: N802
        self.calls.append(("findIndex", forceWaiting, direction))
        return self.index_result

    def setDPOS(
        self,
        value: float,
        differentUnits: object | None = None,
        outputToConsole: bool = True,
        forceWaiting: bool = False,
    ) -> bool | None:  # noqa: N802, E501
        self.calls.append(("setDPOS", value, outputToConsole, forceWaiting))
        return True

    def setSpeed(self, value: float) -> None:  # noqa: N802
        self.calls.append(("setSpeed", value))

    def startScan(
        self, direction: int, execTime: float | None = None, untilLimit: bool = False
    ) -> None:  # noqa: N802, E501
        self.calls.append(("startScan", direction))
        self.flags["scanning"] = True

    def stopScan(self) -> None:  # noqa: N802
        self.calls.append(("stopScan",))
        self.flags["scanning"] = False

    def getData(self, tag: str) -> float | int | str | None:  # noqa: N802
        return self.data[tag]

    def isMotorOn(self) -> bool:
        return self.flags["motor"]  # noqa: N802, E704

    def isClosedLoop(self) -> bool:
        return self.flags["closed"]  # noqa: N802, E704

    def isEncoderAtIndex(self) -> bool:
        return self.flags["index"]  # noqa: N802, E704

    def isEncoderValid(self) -> bool:
        return self.flags["encoder"]  # noqa: N802, E704

    def isPositionReached(self) -> bool:
        return self.flags["reached"]  # noqa: N802, E704

    def isScanning(self) -> bool:
        return self.flags["scanning"]  # noqa: N802, E704

    def isThermalProtection1(self) -> bool:
        return self.flags["thermal1"]  # noqa: N802, E704

    def isThermalProtection2(self) -> bool:
        return self.flags["thermal2"]  # noqa: N802, E704

    def isEncoderError(self) -> bool:
        return self.flags["encoder_error"]  # noqa: N802, E704

    def isAtLeftEnd(self) -> bool:
        return self.flags["left"]  # noqa: N802, E704

    def isAtRightEnd(self) -> bool:
        return self.flags["right"]  # noqa: N802, E704

    def isErrorLimit(self) -> bool:
        return self.flags["error_limit"]  # noqa: N802, E704

    def isSafetyTimeoutTriggered(self) -> bool:
        return self.flags["safety"]  # noqa: N802, E704

    def isPositionFailTriggered(self) -> bool:
        return self.flags["position_fail"]  # noqa: N802, E704


class FakeController:
    def __init__(self, axis: FakeAxis) -> None:
        self.axis = axis
        self.calls: list[tuple[Any, ...]] = []

    def start(
        self,
        external_communication_thread: bool = False,
        external_settings_default: str | None = None,
    ) -> None:  # noqa: E501
        self.calls.append(("start", external_communication_thread, external_settings_default))

    def stop(self) -> None:
        self.calls.append(("stop",))

    def stopMovements(self) -> None:  # noqa: N802
        self.calls.append(("stopMovements",))
        self.axis.flags["scanning"] = False

    def reset(self) -> None:
        self.calls.append(("reset",))

    def setMasterSetting(self, tag: str, value: object, fromSettingsFile: bool = False) -> None:  # noqa: N802, E501
        self.calls.append(("setMasterSetting", tag, value, fromSettingsFile))


def _rig(
    tmp_path: Path,
    *,
    direction_sign: int = 1,
    index_offset_deg: float = 0.0,
) -> tuple[RealGimbal, FakeController, FakeAxis, ManualClock]:
    settings = tmp_path / "settings_default.txt"
    settings.write_text("POLI=97\n", encoding="utf-8")
    cfg = replace(
        GimbalConfig(),
        serial_port="COM7",
        settings_default_path=str(settings),
        direction_sign=direction_sign,
        index_offset_deg=index_offset_deg,
    )
    clock = ManualClock()
    axis = FakeAxis()
    controller = FakeController(axis)
    gimbal = RealGimbal(clock, cfg, factory=cast(Any, lambda *_: (controller, axis)))
    return gimbal, controller, axis, clock


def _initialize(gimbal: RealGimbal) -> None:
    assert isinstance(gimbal.initialize(), Ok)


def test_initialize_loads_settings_overrides_feedback_indexes_and_shutdown(tmp_path: Path) -> None:
    gimbal, controller, axis, _ = _rig(tmp_path)
    _initialize(gimbal)
    assert controller.calls[:3] == [
        ("start", False, str(tmp_path / "settings_default.txt")),
        ("setMasterSetting", "INFO", 4, False),
        ("setMasterSetting", "POLI", 2, False),
    ]
    assert ("setSetting", "LLIM", -10800) in axis.calls
    assert ("setSetting", "HLIM", 10800) in axis.calls
    assert ("findIndex", True, 0) in axis.calls
    assert isinstance(gimbal.shutdown(), Ok)
    assert controller.calls[-2:] == [("stopMovements",), ("stop",)]


def test_initialize_rejects_missing_settings_and_failed_index(tmp_path: Path) -> None:
    missing = RealGimbal(
        ManualClock(),
        replace(
            GimbalConfig(), serial_port="COM7", settings_default_path=str(tmp_path / "missing")
        ),
    )  # noqa: E501
    assert missing.initialize() == Err(FaultCode.GIMBAL_FAULT)
    gimbal, controller, axis, _ = _rig(tmp_path)
    axis.index_result = False
    assert gimbal.initialize() == Err(FaultCode.GIMBAL_FAULT)
    assert ("stopMovements",) in controller.calls
    assert ("stop",) in controller.calls


def test_position_conversion_offset_sign_and_operational_clamp(tmp_path: Path) -> None:
    gimbal, _, axis, _ = _rig(tmp_path, direction_sign=-1, index_offset_deg=5.0)
    _initialize(gimbal)
    assert isinstance(gimbal.set_position(80.0), Ok)
    assert axis.calls[-1] == ("setDPOS", -40.0, False, False)
    gimbal.shutdown()


def test_rate_deadband_quantization_speed_change_and_reversal(tmp_path: Path) -> None:
    gimbal, controller, axis, _ = _rig(tmp_path)
    _initialize(gimbal)
    assert isinstance(gimbal.set_velocity(0.0049), Ok)
    assert ("stopScan",) in axis.calls
    assert isinstance(gimbal.set_velocity(0.015), Ok)
    assert axis.calls[-2:] == [("setSpeed", 0.02), ("startScan", 1)]
    assert isinstance(gimbal.set_velocity(0.034), Ok)
    assert axis.calls[-1] == ("setSpeed", 0.03)
    axis.data["EPOS"] = 5400
    before = len(axis.calls)
    assert isinstance(gimbal.set_velocity(-99.0), Ok)
    assert axis.calls[before:] == [("stopScan",), ("setSpeed", 10.0), ("startScan", -1)]
    assert ("stopMovements",) in controller.calls
    gimbal.shutdown()


def test_read_state_velocity_wrap_and_status_mapping(tmp_path: Path) -> None:
    gimbal, _, axis, clock = _rig(tmp_path, direction_sign=-1, index_offset_deg=5.0)
    _initialize(gimbal)
    axis.data.update(EPOS=-1200, DPOS=-2400, TIME=65530)
    first = gimbal.read_state()
    assert isinstance(first, Ok)
    assert first.value.position_deg == pytest.approx(10.0)
    assert first.value.target_position_deg == pytest.approx(15.0)
    assert first.value.velocity_deg_per_s is None
    clock.advance(0.0012)
    axis.data.update(EPOS=-1224, TIME=6)
    axis.flags["scanning"] = True
    second = gimbal.read_state()
    assert isinstance(second, Ok)
    assert second.value.velocity_deg_per_s == pytest.approx(83.3333333)
    assert second.value.controller_timestamp_s == pytest.approx(6.5542)
    assert second.value.scanning
    gimbal.shutdown()


def test_stale_feedback_and_invalid_status_stop_safely(tmp_path: Path) -> None:
    gimbal, controller, axis, clock = _rig(tmp_path)
    _initialize(gimbal)
    assert isinstance(gimbal.read_state(), Ok)
    clock.advance(0.251)
    assert gimbal.read_state() == Err(FaultCode.GIMBAL_FAULT)
    assert ("stopMovements",) in controller.calls
    axis.data["TIME"] = 1
    axis.flags["encoder"] = False
    assert gimbal.read_state() == Err(FaultCode.GIMBAL_FAULT)
    gimbal.shutdown()


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("motor", False),
        ("closed", False),
        ("encoder", False),
        ("thermal1", True),
        ("thermal2", True),
        ("encoder_error", True),
        ("left", True),
        ("right", True),
        ("error_limit", True),
        ("safety", True),
        ("position_fail", True),
    ],
)
def test_each_invalid_status_maps_to_safe_fault(tmp_path: Path, flag: str, value: bool) -> None:
    gimbal, controller, axis, _ = _rig(tmp_path)
    _initialize(gimbal)
    axis.flags[flag] = value
    assert gimbal.read_state() == Err(FaultCode.GIMBAL_FAULT)
    assert ("stopMovements",) in controller.calls
    gimbal.shutdown()


def test_rate_lease_and_imaging_limit_stop(tmp_path: Path) -> None:
    gimbal, controller, axis, clock = _rig(tmp_path)
    _initialize(gimbal)
    assert isinstance(gimbal.set_velocity(1.0), Ok)
    clock.advance(3.001)
    gimbal.watchdog_tick()
    assert ("stopMovements",) in controller.calls
    axis.data["EPOS"] = 10800
    assert isinstance(gimbal.set_velocity(1.0), Ok)
    assert not axis.flags["scanning"]
    gimbal.shutdown()


def test_duty_fault_latches_but_preserves_stow_reserve(tmp_path: Path) -> None:
    gimbal, _, axis, clock = _rig(tmp_path)
    _initialize(gimbal)
    gimbal.watchdog_tick()
    clock.advance(105.0)
    gimbal.watchdog_tick()
    assert gimbal.set_position(10.0) == Err(FaultCode.GIMBAL_FAULT)
    assert isinstance(gimbal.stow(), Ok)
    assert axis.calls[-1] == ("setDPOS", -45.0, False, False)
    assert isinstance(gimbal.reset_faults(), Ok)
    assert isinstance(gimbal.set_position(10.0), Ok)
    gimbal.shutdown()


def test_vendor_exception_is_mapped_to_fault(tmp_path: Path) -> None:
    gimbal, _, axis, _ = _rig(tmp_path)
    _initialize(gimbal)

    def fail(_: float) -> None:
        raise RuntimeError("vendor failure")

    axis.setSpeed = fail  # type: ignore[assignment]
    assert gimbal.set_velocity(1.0) == Err(FaultCode.GIMBAL_FAULT)
    gimbal.shutdown()
