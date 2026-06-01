"""Real FLIR Blackfly imaging-sensor driver (flight hardware).

PySpin (the FLIR Spinnaker SDK) is imported lazily in __init__ so importing this
module never requires the SDK; only constructing RealSensor does. Frame acquisition
is a stub pending flight-hardware integration; tests and CI use SimSensor.
"""

from flight.libs.messages import RawFrameMsg
from flight.libs.types import Err, FaultCode, Ok, Result


class RealSensor:
    """FLIR Blackfly multispectral camera driver (stub).

    Satisfies the ImagingSensor protocol. Construction requires the optional PySpin
    SDK (the 'camera' extra). Acquisition/tuning are stubs returning safe defaults.
    """

    def __init__(self, serial_number: str | None = None) -> None:
        """Open the FLIR camera via PySpin.

        Args:
            serial_number: Optional camera serial; None selects the first device.

        Raises:
            ImportError: If PySpin is not installed.
        """
        try:
            import PySpin  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PySpin is not installed. Install the FLIR Spinnaker SDK to use "
                "RealSensor; use SimSensor in tests and simulation."
            ) from exc
        self._serial_number = serial_number

    def acquire_frame(self) -> Result[RawFrameMsg, FaultCode]:
        """Capture one frame. Stub pending PySpin GetNextImage integration."""
        return Err(FaultCode.CAMERA_STALL)

    def set_exposure_us(self, exposure: float) -> Result[None, FaultCode]:
        """Set exposure. Stub pending PySpin node-map integration."""
        return Ok(None)

    def set_gain_db(self, gain: float) -> Result[None, FaultCode]:
        """Set gain. Stub pending PySpin node-map integration."""
        return Ok(None)

    def start_acquisition(self) -> Result[None, FaultCode]:
        """Begin acquisition. Stub pending PySpin BeginAcquisition."""
        return Ok(None)

    def stop_acquisition(self) -> Result[None, FaultCode]:
        """Stop acquisition. Stub pending PySpin EndAcquisition."""
        return Ok(None)
