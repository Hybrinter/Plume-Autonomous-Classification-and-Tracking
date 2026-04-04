"""
Camera abstractions for PACT imaging subsystem.

Defines the AbstractCamera Protocol that all camera implementations must satisfy, a stub
implementation for the FLIR Blackfly S hardware camera, and a MockCamera for use in tests.

Design notes:
- PySpin is imported lazily inside FlirBlackflyCamera.__init__() only. Never at module level.
- FlirBlackflyCamera must never be imported in tests. Use MockCamera instead.
- MockCamera is configurable with a list of RawFrameMsg values to return in sequence.
  When the list is exhausted it signals end-of-stream by returning Err(FaultCode.CAMERA_STALL).

Satisfies: REQ-AIML-IMAG-001, REQ-AIML-IMAG-002
"""

from __future__ import annotations

from typing import Optional, Protocol

from pact.types.enums import FaultCode
from pact.types.enums import Err, Ok, Result
from pact.types.messages import RawFrameMsg


class AbstractCamera(Protocol):
    """Protocol that all camera implementations must satisfy.

    Implementations must be thread-safe: acquire_frame() is called from the capture
    thread while set_exposure_us() / set_gain_db() may be called from the control thread.
    """

    def acquire_frame(self) -> "Result[RawFrameMsg, FaultCode]":
        """Capture one raw multispectral frame.

        Returns Ok(RawFrameMsg) on success.
        Returns Err(FaultCode.CAMERA_STALL) if the frame is not available in time.
        """
        ...

    def set_exposure_us(self, exposure: float) -> "Result[None, FaultCode]":
        """Set camera exposure time in microseconds."""
        ...

    def set_gain_db(self, gain: float) -> "Result[None, FaultCode]":
        """Set camera analogue gain in dB."""
        ...

    def start_acquisition(self) -> "Result[None, FaultCode]":
        """Begin continuous frame acquisition."""
        ...

    def stop_acquisition(self) -> "Result[None, FaultCode]":
        """Stop continuous frame acquisition and release buffers."""
        ...


class FlirBlackflyCamera:
    """FLIR Blackfly S BFS-PGE-50S5M-C via PySpin GigE Vision.

    PySpin is an optional dependency — it is NOT on PyPI and must be manually installed
    from the FLIR Spinnaker SDK. PySpin is imported lazily inside __init__() so that the
    module can be loaded on machines without PySpin (e.g., CI runners). Any attempt to
    instantiate this class without PySpin installed will raise ImportError with a clear
    message pointing to the Spinnaker SDK download page.

    Do NOT import this class in tests. Use MockCamera instead.

    # TODO: stub — replace body with real PySpin acquisition calls during hardware
    # integration. Refer to PySpin examples: Acquisition.py, AcquisitionMultipleCamera.py.
    """

    def __init__(self, serial_number: Optional[str] = None) -> None:
        """Initialise PySpin system and open the camera.

        Parameters
        ----------
        serial_number:
            GigE serial number string to select a specific camera. If None, the first
            detected camera is used.

        Raises
        ------
        ImportError
            If PySpin is not installed.
        RuntimeError
            If no cameras are found or the requested serial number is not present.
        """
        try:
            import PySpin  # noqa: F401  # lazy import — PySpin is optional
            self._pyspin = PySpin
        except ImportError as exc:
            raise ImportError(
                "PySpin is not installed. Install the FLIR Spinnaker SDK and its Python "
                "bindings to use FlirBlackflyCamera. Use MockCamera in tests."
            ) from exc

        # TODO: stub — open system, enumerate cameras, select by serial_number
        ...

    def acquire_frame(self) -> "Result[RawFrameMsg, FaultCode]":
        """Capture one frame via PySpin. Stub."""
        # TODO: stub — call cam.GetNextImage(), convert to numpy, build RawFrameMsg
        ...  # type: ignore[return-value]

    def set_exposure_us(self, exposure: float) -> "Result[None, FaultCode]":
        """Set exposure time via PySpin node map. Stub."""
        # TODO: stub — cam.ExposureTime.SetValue(exposure)
        ...  # type: ignore[return-value]

    def set_gain_db(self, gain: float) -> "Result[None, FaultCode]":
        """Set gain via PySpin node map. Stub."""
        # TODO: stub — cam.Gain.SetValue(gain)
        ...  # type: ignore[return-value]

    def start_acquisition(self) -> "Result[None, FaultCode]":
        """Begin continuous acquisition. Stub."""
        # TODO: stub — cam.BeginAcquisition()
        ...  # type: ignore[return-value]

    def stop_acquisition(self) -> "Result[None, FaultCode]":
        """Stop acquisition and release buffers. Stub."""
        # TODO: stub — cam.EndAcquisition(); cam.DeInit()
        ...  # type: ignore[return-value]


class MockCamera:
    """Synthetic camera that returns pre-configured frames for testing.

    No hardware or PySpin required. Satisfies the AbstractCamera Protocol.

    The camera returns frames from `frames` in order. Once the list is exhausted,
    every subsequent call to acquire_frame() returns Err(FaultCode.CAMERA_STALL) to
    signal end-of-stream to the capture loop.

    Parameters
    ----------
    frames:
        Ordered sequence of RawFrameMsg values to return. May be empty (immediately stalls).

    Example
    -------
    >>> from pact.imaging.camera import MockCamera
    >>> cam = MockCamera(frames=[frame_a, frame_b])
    >>> result = cam.acquire_frame()   # Ok(frame_a)
    >>> result = cam.acquire_frame()   # Ok(frame_b)
    >>> result = cam.acquire_frame()   # Err(FaultCode.CAMERA_STALL)
    """

    def __init__(self, frames: list[RawFrameMsg]) -> None:
        self._frames: list[RawFrameMsg] = list(frames)
        self._index: int = 0
        self._acquiring: bool = False

    def acquire_frame(self) -> "Result[RawFrameMsg, FaultCode]":
        """Return the next pre-configured frame, or Err(CAMERA_STALL) if exhausted."""
        if self._index >= len(self._frames):
            return Err(FaultCode.CAMERA_STALL)
        frame = self._frames[self._index]
        self._index += 1
        return Ok(frame)

    def set_exposure_us(self, exposure: float) -> "Result[None, FaultCode]":
        """No-op for MockCamera. Always succeeds."""
        return Ok(None)

    def set_gain_db(self, gain: float) -> "Result[None, FaultCode]":
        """No-op for MockCamera. Always succeeds."""
        return Ok(None)

    def start_acquisition(self) -> "Result[None, FaultCode]":
        """Mark MockCamera as acquiring. Always succeeds."""
        self._acquiring = True
        return Ok(None)

    def stop_acquisition(self) -> "Result[None, FaultCode]":
        """Mark MockCamera as stopped. Always succeeds."""
        self._acquiring = False
        return Ok(None)
