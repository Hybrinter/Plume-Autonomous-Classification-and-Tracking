"""Imaging-sensor hardware abstraction.

Defines the ImagingSensor protocol that every camera driver (real or simulated)
satisfies. Migrated from the original AbstractCamera contract. Implementations
must be thread-safe: acquire_frame() runs on the capture path while the tuning
calls may arrive from a control path.
"""

from typing import Protocol, runtime_checkable

from flight.libs.messages import RawFrameMsg
from flight.libs.types import FaultCode, Result


@runtime_checkable
class ImagingSensor(Protocol):
    """Hardware abstraction for a multispectral imaging sensor."""

    def acquire_frame(self) -> Result[RawFrameMsg, FaultCode]:
        """Capture one raw multispectral frame.

        Returns:
            Result[RawFrameMsg, FaultCode]: Ok(frame) on success;
            Err(FaultCode.CAMERA_STALL) when no frame is available in time.
        """
        ...

    def set_exposure_us(self, exposure: float) -> Result[None, FaultCode]:
        """Set exposure time in microseconds."""
        ...

    def set_gain_db(self, gain: float) -> Result[None, FaultCode]:
        """Set analogue gain in dB."""
        ...

    def start_acquisition(self) -> Result[None, FaultCode]:
        """Begin continuous frame acquisition."""
        ...

    def stop_acquisition(self) -> Result[None, FaultCode]:
        """Stop acquisition and release buffers."""
        ...
