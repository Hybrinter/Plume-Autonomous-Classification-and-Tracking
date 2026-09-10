# ruff: noqa: N802, N803

"""Narrow type-only boundary for the untyped Xeryon v1.88 module.

``Xeryon.py`` is supplied by the manufacturer and is kept byte-for-byte
unchanged.  The protocols below describe only the calls made by PACT flight
software; they avoid leaking the vendor module's large untyped API into the
rest of the workspace.  Importing this module never imports the vendor module
or opens a serial port.
"""

from __future__ import annotations

from typing import Protocol

VendorValue = float | int | str | None


class XeryonAxis(Protocol):
    """Subset of the vendor Axis API used by the real gimbal driver."""

    def findIndex(self, forceWaiting: bool = False, direction: int = 0) -> bool | None: ...

    def setDPOS(
        self,
        value: float,
        differentUnits: object | None = None,
        outputToConsole: bool = True,
        forceWaiting: bool = False,
    ) -> bool | None: ...

    def getDPOS(self) -> float | None: ...

    def getEPOS(self) -> float | None: ...

    def setUnits(self, units: object) -> None: ...

    def setSetting(
        self,
        tag: str,
        value: object,
        fromSettingsFile: bool = False,
        doNotSendThrough: bool = False,
    ) -> None: ...

    def startScan(
        self, direction: int, execTime: float | None = None, untilLimit: bool = False
    ) -> None: ...

    def stopScan(self) -> None: ...

    def setSpeed(self, speed: float) -> None: ...

    def getSetting(self, tag: str) -> VendorValue: ...

    def getData(self, tag: str) -> VendorValue: ...

    def isMotorOn(self, external_stat: int | None = None) -> bool: ...

    def isClosedLoop(self, external_stat: int | None = None) -> bool: ...

    def isEncoderAtIndex(self, external_stat: int | None = None) -> bool: ...

    def isEncoderValid(self, external_stat: int | None = None) -> bool: ...

    def isPositionReached(self, external_stat: int | None = None) -> bool: ...

    def isScanning(self, external_stat: int | None = None) -> bool: ...

    def isEncoderError(self, external_stat: int | None = None) -> bool: ...

    def isAtLeftEnd(self, external_stat: int | None = None) -> bool: ...

    def isAtRightEnd(self, external_stat: int | None = None) -> bool: ...

    def isErrorLimit(self, external_stat: int | None = None) -> bool: ...

    def isThermalProtection1(self, external_stat: int | None = None) -> bool: ...

    def isThermalProtection2(self, external_stat: int | None = None) -> bool: ...

    def isSafetyTimeoutTriggered(self, external_stat: int | None = None) -> bool: ...

    def isPositionFailTriggered(self, external_stat: int | None = None) -> bool: ...


class XeryonController(Protocol):
    """Subset of the vendor Xeryon API used by the real gimbal driver."""

    def start(
        self,
        external_communication_thread: bool = False,
        external_settings_default: str | None = None,
    ) -> None: ...

    def stop(self) -> None: ...

    def stopMovements(self) -> None: ...

    def reset(self) -> None: ...

    def addAxis(self, stage: object, axis_letter: str) -> XeryonAxis: ...

    def getAxis(self, letter: str) -> XeryonAxis: ...

    def readSettings(self, external_settings_default: str | None = None) -> None: ...

    def setMasterSetting(self, tag: str, value: object, fromSettingsFile: bool = False) -> None: ...


__all__ = ["XeryonAxis", "XeryonController"]
