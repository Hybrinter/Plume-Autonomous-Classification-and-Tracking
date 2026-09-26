"""Raw-frame value types exchanged between the imaging HAL and the payload app.

MosaicFrame is NOT a bus message: frames are passed by direct call from the injected
sensor driver to the payload app (co-location invariant; large artifacts never go on
the bus). ``planes`` is three registered RGB images in ``BAND_ORDER``
(BLUE, GREEN, RED), as delivered by the JAI AP-3200T-USB prism. The class name
is historical.

Classes:
- MosaicFrame: frozen dataclass holding raw (3, H, W) planes and the capture
  metadata (timestamp, frame counter, exposure, gain) needed by preprocessing.

Satisfies: REQ-AIML-IMAG-001.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MosaicFrame:
    """One raw frame from the prism camera: three RGB planes plus capture metadata.

    Not a bus message. ``planes`` is np.ndarray (3, H, W) in ``BAND_ORDER``.
    Preprocess calibrates each plane, normalizes, labels quality, then upsamples.

    Inputs:
        timestamp_utc (str): ISO 8601 capture time, millisecond precision.
        timestamp_s (float): Monotonic shutter time in seconds (vehicle clock).
        frame_id (int): Monotonic uint32 frame counter assigned by the driver.
        planes (object): np.ndarray[uint16, (3, H, W)] in BLUE, GREEN, RED order.
            Typed as object to avoid a numpy import at the flight.libs level.
        exposure_us (float): Exposure time in microseconds (shared when the
            channels use one shutter).
        gain_db (float): Analogue gain in dB (shared when the channels match).
        exposure_us_by_band (tuple[float, float, float] | None): Per-channel
            exposure in ``BAND_ORDER``. None means every channel uses exposure_us.
        gain_db_by_band (tuple[float, float, float] | None): Per-channel gain in
            ``BAND_ORDER``. None means every channel uses gain_db.

    Outputs:
        Frozen dataclass instance (immutable after construction).
    """

    timestamp_utc: str
    timestamp_s: float
    frame_id: int
    planes: object  # np.ndarray[uint16, (3, H, W)] in BAND_ORDER
    exposure_us: float
    gain_db: float
    exposure_us_by_band: tuple[float, float, float] | None = None
    gain_db_by_band: tuple[float, float, float] | None = None
