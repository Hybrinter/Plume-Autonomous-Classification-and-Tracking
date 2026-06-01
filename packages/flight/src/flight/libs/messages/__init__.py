"""Inter-subsystem message contract: frozen message dataclasses.

These are the only types that cross the bus between subsystem apps. Import them
from `flight.libs.messages`.
"""

from flight.libs.messages.messages import (
    BlobMeta,
    DownlinkItemMsg,
    FaultEventMsg,
    GimbalCommandMsg,
    HeartbeatMsg,
    InferenceResultMsg,
    ModeChangeMsg,
    ProcessedFrameMsg,
    RawFrameMsg,
    StorageWriteMsg,
    TelemetryEventMsg,
    UploadChunkMsg,
    utc_now_iso,
)

__all__ = [
    "BlobMeta",
    "DownlinkItemMsg",
    "FaultEventMsg",
    "GimbalCommandMsg",
    "HeartbeatMsg",
    "InferenceResultMsg",
    "ModeChangeMsg",
    "ProcessedFrameMsg",
    "RawFrameMsg",
    "StorageWriteMsg",
    "TelemetryEventMsg",
    "UploadChunkMsg",
    "utc_now_iso",
]
