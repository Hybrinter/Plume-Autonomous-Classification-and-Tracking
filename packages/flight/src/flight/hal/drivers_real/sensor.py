"""Real FLIR Blackfly imaging-sensor driver (flight hardware).

PySpin (the FLIR Spinnaker SDK) is imported lazily in __init__ so importing this
module never requires the SDK; only constructing RealSensor does. The driver ACQUIRES
ONLY -- a raw uint16 mosaic plane out, no image processing (raw-mosaic ingest contract).
Frame acquisition is a stub returning Err(CAMERA_STALL) pending the full PySpin
integration (added in a later task); tests and CI use SimSensor.

Contains:
  - RealSensor: lazy-PySpin camera driver satisfying ImagingSensor structurally (stub
    acquisition; full node-map control plane lands with the PySpin integration).

Satisfies: REQ-AIML-IMAG-001.
"""

from flight.libs.types import Err, FaultCode, MosaicFrame, Ok, Result


class RealSensor:
    """FLIR Blackfly mosaic camera driver (stub).

    Satisfies the ImagingSensor protocol. Construction requires the optional PySpin
    SDK (the 'camera' extra). Acquisition/tuning are stubs returning safe defaults
    until the PySpin acquisition + node-map integration is completed.
    """

    def __init__(self, serial_number: str | None = None) -> None:
        """Open the FLIR camera via PySpin.

        Inputs:
            serial_number (str | None): Optional camera serial; None selects the first
                device.

        Outputs:
            None.

        Raises:
            ImportError: If PySpin (the FLIR Spinnaker SDK) is not installed.
        """
        try:
            import PySpin  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PySpin is not installed. Install the FLIR Spinnaker SDK to use "
                "RealSensor; use SimSensor in tests and simulation."
            ) from exc
        self._serial_number = serial_number

    def acquire_frame(self) -> Result[MosaicFrame, FaultCode]:
        """Capture one raw mosaic frame. Stub pending PySpin GetNextImage integration.

        Inputs:
            None.

        Returns:
            Result[MosaicFrame, FaultCode]: Always Err(FaultCode.CAMERA_STALL) in this
            stub; the real implementation returns Ok(MosaicFrame) on a complete capture.
        """
        return Err(FaultCode.CAMERA_STALL)

    def set_exposure_us(self, exposure: float) -> Result[None, FaultCode]:
        """Set exposure. Stub pending PySpin node-map integration.

        Inputs:
            exposure (float): Exposure time in microseconds (ignored in this stub).

        Returns:
            Result[None, FaultCode]: Ok(None) always.
        """
        return Ok(None)

    def set_gain_db(self, gain: float) -> Result[None, FaultCode]:
        """Set gain. Stub pending PySpin node-map integration.

        Inputs:
            gain (float): Analogue gain in dB (ignored in this stub).

        Returns:
            Result[None, FaultCode]: Ok(None) always.
        """
        return Ok(None)

    def start_acquisition(self) -> Result[None, FaultCode]:
        """Begin acquisition. Stub pending PySpin BeginAcquisition.

        Inputs:
            None.

        Returns:
            Result[None, FaultCode]: Ok(None) always.
        """
        return Ok(None)

    def stop_acquisition(self) -> Result[None, FaultCode]:
        """Stop acquisition. Stub pending PySpin EndAcquisition.

        Inputs:
            None.

        Returns:
            Result[None, FaultCode]: Ok(None) always.
        """
        return Ok(None)
