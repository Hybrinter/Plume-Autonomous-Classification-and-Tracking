"""Raw-frame value types exchanged between the imaging HAL and the payload app.

MosaicFrame is NOT a bus message: frames are passed by direct call from the injected
sensor driver to the payload app (co-location invariant; large artifacts never go on
the bus). The buffer is the prism camera delivery, (3, H, W) or (H, W, 3), exactly
as read from the sensor. Preprocessing stacks it once.

Classes:
- MosaicFrame: frozen dataclass holding a raw (H, W) uint16 CFA mosaic plane and
  the capture metadata (timestamp, frame counter, exposure, gain) needed by the
  preprocessing and quality-gate pipeline.

Satisfies: REQ-AIML-IMAG-001.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MosaicFrame:
    """One raw frame from the imaging sensor: prism buffer + capture metadata.

    Not a bus message -- frames are passed by direct function call from the sensor
    driver to the payload app (co-location invariant). The mosaic field holds the
    raw camera buffer; stacking, bad-pixel repair, dark/flat correction, and
    normalization run in the payload app pipeline after construction.

    Inputs:
        timestamp_utc (str): ISO 8601 capture time, millisecond precision,
            e.g. "2026-06-09T12:00:00.000Z".
        timestamp_s (float): Monotonic shutter time in seconds (vehicle clock).
        frame_id (int): Monotonic uint32 frame counter assigned by the driver.
        mosaic (object): np.ndarray camera buffer, (3, H, W) or (H, W, 3).
            Typed as object to avoid a numpy import at the flight.libs level.
        exposure_us (float): Exposure time in microseconds.
        gain_db (float): Analogue gain in dB.

    Outputs:
        Frozen dataclass instance (immutable after construction).

    Notes:
        The mosaic field is typed as object rather than np.ndarray to avoid
        pulling numpy into flight.libs.types (a dependency root). Callers
        retrieve the array via np.asarray(frame.mosaic).
    """

    timestamp_utc: str
    timestamp_s: float
    frame_id: int
    mosaic: object  # np.ndarray, (3, H, W) or (H, W, 3)
    exposure_us: float
    gain_db: float
