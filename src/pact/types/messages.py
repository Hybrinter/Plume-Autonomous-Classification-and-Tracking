"""PACT inter-process message dataclasses. §4.2 of PACT_SW_ARCH.

All messages passed across concurrency boundaries (multiprocessing.Queue,
queue.Queue, or asyncio.Queue) are defined here. Every message:
  - Is @dataclass(frozen=True) to mirror Rust structs.
  - Has msg_type: MessageType as its first field (discriminant).
  - Has timestamp_utc: str (ISO 8601, millisecond precision) as its second field.
  - Has frame_id: int where applicable (uint32, monotonic counter).

No other pact submodule except pact.types.enums is imported here.

Satisfies: REQ-AIML-COMP-001, REQ-AIML-COMP-002 (typed message-passing between processes).
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass

# third-party
import numpy as np  # noqa: F401  (used in type comments)

# internal
from pact.types.enums import (
    DownlinkPriority,
    FaultCode,
    FrameUsabilityTag,
    GimbalState,
    MessageType,
    SystemMode,
)


# ---------------------------------------------------------------------------
# Embedded sub-structs (not messages themselves, but embedded in messages)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlobMeta:
    """Metadata for a single detected plume blob.

    Embedded in InferenceResultMsg. blob_id is a uint16 persistent tracker ID
    assigned by controller/tracker.py and inherited across frames via IoU matching.
    """

    blob_id: int                            # uint16 persistent tracker ID
    bbox: tuple[int, int, int, int]         # (x_min, y_min, x_max, y_max) pixel space
    centroid_raw: tuple[float, float]       # (x, y) float centroid in crop-space pixels
    pixel_area: int                         # number of pixels in blob mask
    mean_confidence: float                  # mean softmax probability over blob pixels
    persistence_count: int                  # consecutive frames this blob has been tracked


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawFrameMsg:
    """Raw multispectral frame from imaging to preprocessing.

    raw_bands shape: (C, H, W) float32, where C = number of camera bands.
    """

    msg_type: MessageType                   # must be MessageType.RAW_FRAME
    timestamp_utc: str                      # ISO 8601, millisecond precision
    frame_id: int                           # uint32 monotonic frame counter
    raw_bands: object                       # np.ndarray[float32, (C, H, W)]
    exposure_us: float                      # camera exposure time in microseconds
    gain_db: float                          # camera gain in dB
    gimbal_az_deg: float                    # gimbal azimuth at capture time (degrees)
    gimbal_el_deg: float                    # gimbal elevation at capture time (degrees)


@dataclass(frozen=True)
class ProcessedFrameMsg:
    """Preprocessed, band-selected, calibrated tensor from preprocessing to inference.

    tensor shape: (4, H, W) float32, bands B2/B3/B4/B8 in that order.
    """

    msg_type: MessageType                   # must be MessageType.PROCESSED_FRAME
    timestamp_utc: str                      # ISO 8601, millisecond precision
    frame_id: int                           # uint32 monotonic frame counter
    tensor: object                          # np.ndarray[float32, (4, H, W)]
    quality_flags: frozenset[FrameUsabilityTag]
    crop_origin_px: tuple[int, int]         # (x, y) top-left offset of crop in full frame
    scale_factor: float                     # resize scale applied during preprocessing


@dataclass(frozen=True)
class InferenceResultMsg:
    """Segmentation output from inference to controller and storage.

    mask shape: (H, W) float32, softmax probability per pixel.
    """

    msg_type: MessageType                   # must be MessageType.INFERENCE_RESULT
    timestamp_utc: str                      # ISO 8601, millisecond precision
    frame_id: int                           # uint32 monotonic frame counter
    mask: object                            # np.ndarray[float32, (H, W)]
    blobs: tuple[BlobMeta, ...]             # zero or more detected blobs
    model_version: str                      # model checkpoint identifier string
    inference_ms: float                     # wall-clock inference duration in ms
    mode_flags: int                         # uint8 bitmask; semantics defined in config


@dataclass(frozen=True)
class GimbalCommandMsg:
    """Gimbal slew command from controller to gimbal hardware interface."""

    msg_type: MessageType                   # must be MessageType.GIMBAL_COMMAND
    timestamp_utc: str                      # ISO 8601, millisecond precision
    frame_id: int                           # frame that triggered this command
    az_delta_deg: float                     # azimuth delta to command (degrees)
    el_delta_deg: float                     # elevation delta to command (degrees)
    state: GimbalState                      # arbiter state at time of command
    reason: str                             # human-readable reason code for logging


@dataclass(frozen=True)
class TelemetryEventMsg:
    """Structured telemetry event from any subsystem to the telemetry reporter.

    payload must contain only JSON-serializable primitive types.
    """

    msg_type: MessageType                   # must be MessageType.TELEMETRY_EVENT
    timestamp_utc: str                      # ISO 8601, millisecond precision
    subsystem: str                          # originating subsystem name (snake_case)
    event_name: str                         # short snake_case event identifier
    payload: dict[str, str | int | float | bool]  # serializable structured fields only


@dataclass(frozen=True)
class FaultEventMsg:
    """Fault notification from any subsystem to the fault detection process."""

    msg_type: MessageType                   # must be MessageType.FAULT_EVENT
    timestamp_utc: str                      # ISO 8601, millisecond precision
    fault_code: FaultCode                   # enumerated fault condition
    subsystem: str                          # subsystem that detected/raised the fault
    detail: str                             # human-readable fault detail string


@dataclass(frozen=True)
class HeartbeatMsg:
    """Periodic liveness signal from each subsystem to the fault watchdog."""

    msg_type: MessageType                   # must be MessageType.HEARTBEAT
    timestamp_utc: str                      # ISO 8601, millisecond precision
    subsystem: str                          # originating subsystem name
    sequence: int                           # monotonic heartbeat counter per subsystem


@dataclass(frozen=True)
class ModeChangeMsg:
    """System mode transition request or notification."""

    msg_type: MessageType                   # must be MessageType.MODE_CHANGE
    timestamp_utc: str                      # ISO 8601, millisecond precision
    new_mode: SystemMode                    # requested target system mode
    requested_by: str                       # subsystem or operator that requested the change


@dataclass(frozen=True)
class StorageWriteMsg:
    """Bundle of a full frame's data for the storage writer process.

    raw_frame shape: (C, H, W) float32.
    processed_tensor shape: (4, H, W) float32.
    """

    msg_type: MessageType                   # must be MessageType.STORAGE_WRITE
    timestamp_utc: str                      # ISO 8601, millisecond precision
    frame_id: int                           # uint32 monotonic frame counter
    raw_frame: object                       # np.ndarray[float32, (C, H, W)]
    processed_tensor: object                # np.ndarray[float32, (4, H, W)]
    inference_result: InferenceResultMsg    # full inference output for this frame
    usability: FrameUsabilityTag            # computed usability classification


@dataclass(frozen=True)
class DownlinkItemMsg:
    """A single item queued for CCSDS downlink, with priority and CRC."""

    msg_type: MessageType                   # must be MessageType.DOWNLINK_ITEM
    timestamp_utc: str                      # ISO 8601, millisecond precision
    priority: DownlinkPriority              # queue priority (lower int == higher priority)
    payload_bytes: bytes                    # serialized content to downlink
    crc32: int                              # CRC-32 of payload_bytes
    item_id: str                            # unique item identifier string


@dataclass(frozen=True)
class UploadChunkMsg:
    """One chunk of a chunked model upload via CCSDS uplink.

    Required by comms/uplink.py. Added per §8 known-gap note in types/CLAUDE.md.
    chunk_index is zero-based. data contains the raw bytes for this chunk.
    expected_crc32 is the CRC-32 of the complete reassembled model file (not this chunk).
    """

    msg_type: MessageType                   # must be MessageType.UPLINK_CHUNK
    timestamp_utc: str                      # ISO 8601, millisecond precision
    chunk_index: int                        # zero-based chunk index
    total_chunks: int                       # total number of chunks in this upload session
    data: bytes                             # raw bytes for this chunk
    expected_crc32: int                     # CRC-32 of the complete reassembled file
