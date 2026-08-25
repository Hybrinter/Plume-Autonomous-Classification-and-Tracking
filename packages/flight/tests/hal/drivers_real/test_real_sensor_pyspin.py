"""RealSensor behavior tests against a fake PySpin module (no SDK in CI)."""

import sys
import types

import numpy as np
import pytest
from flight.libs.time import ManualClock
from flight.libs.types import Err, FaultCode, Ok


class _FakeImage:
    def __init__(self, incomplete: bool = False) -> None:
        self._incomplete = incomplete

    def IsIncomplete(self) -> bool:  # noqa: N802 - PySpin API casing
        return self._incomplete

    def GetNDArray(self) -> np.ndarray:  # noqa: N802
        return np.full((4, 4), 100, dtype=np.uint16)

    def Release(self) -> None:  # noqa: N802
        pass


class _FakeFloatNode:
    def __init__(self, value: float) -> None:
        self._value = value

    def SetValue(self, value: float) -> None:  # noqa: N802
        self._value = value

    def GetValue(self) -> float:  # noqa: N802
        return self._value


class _FakeCamera:
    def __init__(self) -> None:
        self.ExposureTime = _FakeFloatNode(1000.0)
        self.Gain = _FakeFloatNode(0.0)
        self.next_image: _FakeImage | Exception = _FakeImage()

    def Init(self) -> None:  # noqa: N802
        pass

    def BeginAcquisition(self) -> None:  # noqa: N802
        pass

    def EndAcquisition(self) -> None:  # noqa: N802
        pass

    def GetNextImage(self, timeout_ms: int) -> _FakeImage:  # noqa: N802
        if isinstance(self.next_image, Exception):
            raise self.next_image
        return self.next_image


def _install_fake_pyspin(monkeypatch: pytest.MonkeyPatch, camera: _FakeCamera) -> None:
    fake = types.ModuleType("PySpin")

    class SpinnakerException(Exception):  # noqa: N818 - mirrors the PySpin SDK name
        pass

    class _CamList:
        def GetSize(self) -> int:  # noqa: N802
            return 1

        def GetByIndex(self, index: int) -> _FakeCamera:  # noqa: N802
            return camera

        def GetBySerial(self, serial: str) -> _FakeCamera:  # noqa: N802
            return camera

    class _System:
        @staticmethod
        def GetInstance() -> _System:  # noqa: N802
            return _System()

        def GetCameras(self) -> _CamList:  # noqa: N802
            return _CamList()

    fake.SpinnakerException = SpinnakerException  # type: ignore[attr-defined]
    fake.System = _System  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PySpin", fake)


def test_acquire_frame_returns_mosaic(monkeypatch: pytest.MonkeyPatch) -> None:
    """A complete image converts to a uint16 MosaicFrame with metadata."""
    from flight.hal.drivers_real import RealSensor

    camera = _FakeCamera()
    _install_fake_pyspin(monkeypatch, camera)
    sensor = RealSensor(clock=ManualClock())
    result = sensor.acquire_frame()
    assert isinstance(result, Ok)
    assert np.asarray(result.value.mosaic).dtype == np.uint16
    assert result.value.frame_id == 1
    assert result.value.exposure_us == 1000.0


def test_incomplete_image_is_camera_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    """An incomplete transfer returns Err(CAMERA_STALL)."""
    from flight.hal.drivers_real import RealSensor

    camera = _FakeCamera()
    camera.next_image = _FakeImage(incomplete=True)
    _install_fake_pyspin(monkeypatch, camera)
    sensor = RealSensor(clock=ManualClock())
    result = sensor.acquire_frame()
    assert isinstance(result, Err)
    assert result.error == FaultCode.CAMERA_STALL


def test_sdk_timeout_is_camera_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    """A SpinnakerException during GetNextImage returns Err(CAMERA_STALL)."""
    from flight.hal.drivers_real import RealSensor

    camera = _FakeCamera()
    _install_fake_pyspin(monkeypatch, camera)
    sensor = RealSensor(clock=ManualClock())
    camera.next_image = sys.modules["PySpin"].SpinnakerException("timeout")
    result = sensor.acquire_frame()
    assert isinstance(result, Err)
    assert result.error == FaultCode.CAMERA_STALL


def test_set_exposure_writes_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """set_exposure_us writes the camera ExposureTime node."""
    from flight.hal.drivers_real import RealSensor

    camera = _FakeCamera()
    _install_fake_pyspin(monkeypatch, camera)
    sensor = RealSensor(clock=ManualClock())
    assert isinstance(sensor.set_exposure_us(2500.0), Ok)
    assert camera.ExposureTime.GetValue() == 2500.0
