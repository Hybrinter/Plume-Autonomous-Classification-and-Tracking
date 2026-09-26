"""Real imaging-sensor driver over PySpin.

PySpin (the FLIR Spinnaker SDK) imports lazily in __init__: importing this module never
needs the SDK, only constructing RealSensor does. The driver acquires only: three
uint16 planes in BAND_ORDER, with no calibration or normalization. acquire_frame and
the control plane (exposure/gain/start/stop) are
serialized with a single lock so the capture loop and tuning commands can run from
different threads. A failed or incomplete transfer maps to Err(CAMERA_STALL); the
payload loop degrades gracefully rather than raising.

Contains:
  - RealSensor: lazy-PySpin camera driver satisfying the ImagingSensor
    Protocol structurally. Captures one raw RGB frame per acquire_frame, exposes the
    ExposureTime/Gain node-map control plane, and brackets acquisition with
    Begin/EndAcquisition. All node access is lock-serialized.

Satisfies: REQ-AIML-IMAG-001.
"""

from __future__ import annotations

# stdlib
import threading

# third-party
import numpy as np

# internal
from flight.libs.time import Clock
from flight.libs.types import Err, FaultCode, MosaicFrame, Ok, Result


class RealSensor:
    """PySpin camera driver, satisfying ImagingSensor structurally.

    Wraps a single PySpin camera handle. Acquisition returns three raw uint16 planes
    in BAND_ORDER. Calibration and normalization happen downstream in
    flight.payload.preprocess. Every method is serialized on an internal lock so the
    acquisition thread and the control plane do not race on the node map.

    Notes:
        Construction requires the FLIR Spinnaker SDK (PySpin), imported lazily so this
        module and its tests load without the SDK. frame_id is a driver-assigned uint32
        monotonic counter starting at 1 for the first captured frame.
    """

    def __init__(
        self,
        clock: Clock,
        serial_number: str | None = None,
        timeout_ms: int = 1000,
    ) -> None:
        """Open the camera via PySpin and initialize it.

        Inputs:
            clock (Clock): Injected time source; wall_clock_iso() stamps each frame.
            serial_number (str | None): Optional camera serial; None selects the first
                enumerated device (index 0).
            timeout_ms (int): GetNextImage timeout in milliseconds.

        Outputs:
            None.

        Raises:
            ImportError: If PySpin (the FLIR Spinnaker SDK) is not installed.

        Notes:
            The PySpin System singleton and camera handle are retained for the driver's
            lifetime; the camera is Init()'d here but acquisition is not begun until
            start_acquisition() is called.
        """
        try:
            import PySpin
        except ImportError as exc:
            raise ImportError(
                "PySpin is not installed. Install the FLIR Spinnaker SDK to use "
                "RealSensor; use SimSensor in tests and simulation."
            ) from exc
        self._pyspin = PySpin
        self._system = PySpin.System.GetInstance()
        cameras = self._system.GetCameras()
        self._cam = cameras.GetBySerial(serial_number) if serial_number else cameras.GetByIndex(0)
        self._cam.Init()
        self._clock = clock
        self._timeout_ms = timeout_ms
        self._frame_id = 0
        self._lock = threading.Lock()

    def acquire_frame(self) -> Result[MosaicFrame, FaultCode]:
        """Capture one raw RGB frame from the camera.

        Inputs:
            None.

        Returns:
            Result[MosaicFrame, FaultCode]: Ok(MosaicFrame) carrying uint16 planes of
            shape (3, H, W) in BAND_ORDER plus timestamp/exposure/gain metadata on a
            complete transfer; Err(FaultCode.CAMERA_STALL) on an SDK timeout, an
            incomplete image, or a buffer that is not three channels.

        Notes:
            The PySpin image buffer is always Release()'d (including on the incomplete
            path) before returning so the SDK buffer pool is not exhausted. The buffer
            is copied out of the SDK (copy=True) so the returned array outlives the
            released buffer. A packed (H, W, 3) buffer is moved to (3, H, W). Packed
            channel 0 is blue, which matches BAND_ORDER. frame_id increments only on
            a successful capture.
        """
        with self._lock:
            try:
                image = self._cam.GetNextImage(self._timeout_ms)
            except self._pyspin.SpinnakerException:
                return Err(FaultCode.CAMERA_STALL)
            if image.IsIncomplete():
                image.Release()
                return Err(FaultCode.CAMERA_STALL)
            raw = np.array(image.GetNDArray(), dtype=np.uint16, copy=True)
            image.Release()
            # USB3 Vision RGB is packed B, G, R. That is BAND_ORDER.
            if raw.ndim == 3 and raw.shape[2] == 3:
                planes = np.moveaxis(raw, 2, 0)
            elif raw.ndim == 3 and raw.shape[0] == 3:
                planes = raw
            else:
                return Err(FaultCode.CAMERA_STALL)
            self._frame_id += 1
            return Ok(
                MosaicFrame(
                    timestamp_utc=self._clock.wall_clock_iso(),
                    timestamp_s=self._clock.monotonic_s(),
                    frame_id=self._frame_id,
                    planes=planes,
                    exposure_us=float(self._cam.ExposureTime.GetValue()),
                    gain_db=float(self._cam.Gain.GetValue()),
                )
            )

    def set_exposure_us(self, exposure: float) -> Result[None, FaultCode]:
        """Write the camera ExposureTime node.

        Inputs:
            exposure (float): Exposure time in microseconds.

        Returns:
            Result[None, FaultCode]: Ok(None) on success; Err(FaultCode.CAMERA_STALL)
            on an SDK error.
        """
        with self._lock:
            try:
                self._cam.ExposureTime.SetValue(exposure)
            except self._pyspin.SpinnakerException:
                return Err(FaultCode.CAMERA_STALL)
            return Ok(None)

    def set_gain_db(self, gain: float) -> Result[None, FaultCode]:
        """Write the camera Gain node.

        Inputs:
            gain (float): Analogue gain in dB.

        Returns:
            Result[None, FaultCode]: Ok(None) on success; Err(FaultCode.CAMERA_STALL)
            on an SDK error.
        """
        with self._lock:
            try:
                self._cam.Gain.SetValue(gain)
            except self._pyspin.SpinnakerException:
                return Err(FaultCode.CAMERA_STALL)
            return Ok(None)

    def start_acquisition(self) -> Result[None, FaultCode]:
        """Begin streaming acquisition on the camera (BeginAcquisition).

        Inputs:
            None.

        Returns:
            Result[None, FaultCode]: Ok(None) on success; Err(FaultCode.CAMERA_STALL)
            on an SDK error.
        """
        with self._lock:
            try:
                self._cam.BeginAcquisition()
            except self._pyspin.SpinnakerException:
                return Err(FaultCode.CAMERA_STALL)
            return Ok(None)

    def stop_acquisition(self) -> Result[None, FaultCode]:
        """End streaming acquisition on the camera (EndAcquisition).

        Inputs:
            None.

        Returns:
            Result[None, FaultCode]: Ok(None) on success; Err(FaultCode.CAMERA_STALL)
            on an SDK error.
        """
        with self._lock:
            try:
                self._cam.EndAcquisition()
            except self._pyspin.SpinnakerException:
                return Err(FaultCode.CAMERA_STALL)
            return Ok(None)
