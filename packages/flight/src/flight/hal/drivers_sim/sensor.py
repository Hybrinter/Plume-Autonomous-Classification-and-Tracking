"""Simulated imaging sensor.

Replays a fixed list of raw MosaicFrame frames in order, returning Err(CAMERA_STALL)
once exhausted (matching real stall semantics). Satisfies the ImagingSensor protocol
structurally. Frames are rendered by sim.scene; the driver itself does no image
processing (acquire-only contract).

load_next is a sim-only single-slot mutator (not on ImagingSensor). A closed-loop
bind may overwrite the unread slot each step. Empty constructor plus no slot stalls.
unread_scripted_count reports remaining constructor frames after the live slot.

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
                acquire_frame() call, in order. An empty list stalls until
                load_next supplies a slot.

        Outputs:
            None.
        """
        self._frames = frames
        self._index = 0
        self._slot: MosaicFrame | None = None
        self._acquiring = False

    def load_next(self, frame: MosaicFrame) -> None:
        """Overwrite the single unread live slot.

        Not on ImagingSensor. An unread slot is discarded when this is called
        again. acquire_frame prefers the slot over remaining constructor frames.
        Callers must not overlap load_next with acquire_frame. The slot has no lock.

        Args:
            frame: Mosaic to return on the next acquire_frame.
        """
        self._slot = frame

    def unread_scripted_count(self) -> int:
        """Return how many constructor frames remain after the live slot.

        Not on ImagingSensor. A SIL bind uses this to refuse live mosaics while
        scripted frames remain, so frame_id values stay unique.

        Returns:
            Remaining constructor frames (0 when the list is exhausted).
        """
        return max(0, len(self._frames) - self._index)

    def acquire_frame(self) -> Result[MosaicFrame, FaultCode]:
        """Return the live slot, else the next scripted frame, else CAMERA_STALL.

        Inputs:
            None.

        Returns:
            Result[MosaicFrame, FaultCode]: Ok(frame) while a slot or scripted
            frame remains; Err(FaultCode.CAMERA_STALL) when both are empty.
        """
        if self._slot is not None:
            frame = self._slot
            self._slot = None
            return Ok(frame)
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
