"""Simulated imaging sensor.

Replays a fixed list of frames in order, returning Err(CAMERA_STALL) once exhausted
(matching real stall semantics). Satisfies the ImagingSensor protocol structurally.
Used by SIL; later it can be fed by the scene generator.
"""

from flight.libs.messages import RawFrameMsg
from flight.libs.types import Err, FaultCode, Ok, Result


class SimSensor:
    """Imaging sensor that returns pre-loaded frames in order (sim/SIL driver)."""

    def __init__(self, frames: list[RawFrameMsg]) -> None:
        """Initialize with the ordered frames to replay.

        Args:
            frames: Frames returned one per acquire_frame() call, in order.
        """
        self._frames = frames
        self._index = 0
        self._acquiring = False

    def acquire_frame(self) -> Result[RawFrameMsg, FaultCode]:
        """Return the next frame, or Err(CAMERA_STALL) once exhausted."""
        if self._index >= len(self._frames):
            return Err(FaultCode.CAMERA_STALL)
        frame = self._frames[self._index]
        self._index += 1
        return Ok(frame)

    def set_exposure_us(self, exposure: float) -> Result[None, FaultCode]:
        """No-op for the simulated sensor."""
        return Ok(None)

    def set_gain_db(self, gain: float) -> Result[None, FaultCode]:
        """No-op for the simulated sensor."""
        return Ok(None)

    def start_acquisition(self) -> Result[None, FaultCode]:
        """Mark the simulated sensor as acquiring."""
        self._acquiring = True
        return Ok(None)

    def stop_acquisition(self) -> Result[None, FaultCode]:
        """Mark the simulated sensor as stopped."""
        self._acquiring = False
        return Ok(None)
