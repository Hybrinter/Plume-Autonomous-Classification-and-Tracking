"""Inter-subsystem message contract: frozen message dataclasses.

These are the only types that cross the bus between subsystem apps. Import them
from `flight.libs.messages`.
"""

from flight.libs.messages.messages import (
    BlobMeta,
    CommandMsg,
    DownlinkItemMsg,
    FaultEventMsg,
    GimbalCommandMsg,
    HeartbeatMsg,
    InferenceResultMsg,
    ModeChangeMsg,
    ProcessedFrameMsg,
    StorageWriteMsg,
    TelemetryEventMsg,
    UploadChunkMsg,
    utc_now_iso,
)

__all__ = [
    "BlobMeta",
    "CommandMsg",
    "DownlinkItemMsg",
    "FaultEventMsg",
    "GimbalCommandMsg",
    "HeartbeatMsg",
    "InferenceResultMsg",
    "ModeChangeMsg",
    "ProcessedFrameMsg",
    "StorageWriteMsg",
    "TelemetryEventMsg",
    "UploadChunkMsg",
    "utc_now_iso",
]
