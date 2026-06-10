"""Simulated imaging sensor.

Replays a fixed list of raw MosaicFrame frames in order, returning Err(CAMERA_STALL)
once exhausted (matching real stall semantics). Satisfies the ImagingSensor protocol
structurally. Frames are rendered by sim.scene; the driver itself does no image
processing (acquire-only contract).

Contains:
  - SimSensor: replays pre-loaded MosaicFrame frames one per acquire_frame() call.
"""

from flight.libs.types import Err, FaultCode, MosaicFrame, Ok, Result


class SimSensor:
    """Imaging sensor that returns pre-loaded mosaic frames in order (sim/SIL driver)."""

    def __init__(self, frames: list[MosaicFrame]) -> None:
        """Initialize with the ordered frames to replay.

        Inputs:
            frames (list[MosaicFrame]): Raw mosaic frames returned one per
                acquire_frame() call, in order.

        Outputs:
            None.
        """
        self._frames = frames
        self._index = 0
        self._acquiring = False

    def acquire_frame(self) -> Result[MosaicFrame, FaultCode]:
        """Return the next mosaic frame, or Err(CAMERA_STALL) once exhausted.

        Inputs:
            None.

        Returns:
            Result[MosaicFrame, FaultCode]: Ok(frame) while frames remain;
            Err(FaultCode.CAMERA_STALL) after the replay list is exhausted.
        """
        if self._index >= len(self._frames):
            return Err(FaultCode.CAMERA_STALL)
        frame = self._frames[self._index]
        self._index += 1
        return Ok(frame)

    def set_exposure_us(self, exposure: float) -> Result[None, FaultCode]:
        """No-op for the simulated sensor.

        Inputs:
            exposure (float): Exposure time in microseconds (ignored).

        Returns:
            Result[None, FaultCode]: Ok(None) always.
        """
        return Ok(None)

    def set_gain_db(self, gain: float) -> Result[None, FaultCode]:
        """No-op for the simulated sensor.

        Inputs:
            gain (float): Analogue gain in dB (ignored).

        Returns:
            Result[None, FaultCode]: Ok(None) always.
        """
        return Ok(None)

    def start_acquisition(self) -> Result[None, FaultCode]:
        """Mark the simulated sensor as acquiring.

        Inputs:
            None.

        Returns:
            Result[None, FaultCode]: Ok(None) always.
        """
        self._acquiring = True
        return Ok(None)

    def stop_acquisition(self) -> Result[None, FaultCode]:
        """Mark the simulated sensor as stopped.

        Inputs:
            None.

        Returns:
            Result[None, FaultCode]: Ok(None) always.
        """
        self._acquiring = False
        return Ok(None)
