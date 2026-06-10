"""Imaging-sensor hardware abstraction.

Defines the ImagingSensor protocol that every camera driver (real or simulated)
satisfies. Drivers ACQUIRE ONLY: acquire_frame() returns a raw (H, W) uint16 2x2-CFA
mosaic plane (a MosaicFrame), with NO demosaic, calibration, or normalization inside
any driver (ADR: raw-mosaic ingest contract). Those stages run as pure functions in
flight.payload.preprocess. Implementations must be thread-safe: acquire_frame() runs on
the capture path while the tuning calls may arrive from a control path.

Contains:
  - ImagingSensor: the runtime-checkable Protocol every camera driver satisfies.

Satisfies: REQ-AIML-IMAG-001.
"""

from typing import Protocol, runtime_checkable

from flight.libs.types import FaultCode, MosaicFrame, Result


@runtime_checkable
class ImagingSensor(Protocol):
    """Hardware abstraction for a 2x2-mosaic imaging sensor (acquire-only contract).

    Implementations return a raw MosaicFrame from acquire_frame() and never perform
    image processing; demosaic/calibration/normalization live in preprocess. The
    control-plane methods (exposure/gain, acquisition start/stop) stay on the Protocol.
    """

    def acquire_frame(self) -> Result[MosaicFrame, FaultCode]:
        """Capture one raw 2x2-CFA mosaic frame.

        Inputs:
            None.

        Returns:
            Result[MosaicFrame, FaultCode]: Ok(frame) carrying the raw (H, W) uint16
            mosaic plane plus capture metadata on success; Err(FaultCode.CAMERA_STALL)
            when no complete frame is available in time.

        Notes:
            The returned plane is un-demosaicked: it is the raw CFA image straight off
            the sensor. No calibration or normalization has been applied.
        """
        ...

    def set_exposure_us(self, exposure: float) -> Result[None, FaultCode]:
        """Set exposure time in microseconds.

        Inputs:
            exposure (float): Exposure time in microseconds.

        Returns:
            Result[None, FaultCode]: Ok(None) on success; Err on an SDK/control error.
        """
        ...

    def set_gain_db(self, gain: float) -> Result[None, FaultCode]:
        """Set analogue gain in dB.

        Inputs:
            gain (float): Analogue gain in dB.

        Returns:
            Result[None, FaultCode]: Ok(None) on success; Err on an SDK/control error.
        """
        ...

    def start_acquisition(self) -> Result[None, FaultCode]:
        """Begin continuous frame acquisition.

        Inputs:
            None.

        Returns:
            Result[None, FaultCode]: Ok(None) on success; Err on an SDK/control error.
        """
        ...

    def stop_acquisition(self) -> Result[None, FaultCode]:
        """Stop acquisition and release buffers.

        Inputs:
            None.

        Returns:
            Result[None, FaultCode]: Ok(None) on success; Err on an SDK/control error.
        """
        ...
