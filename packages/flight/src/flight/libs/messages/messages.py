"""Flight inter-subsystem message dataclasses.

All messages published on the in-process MessageBus (queue.Queue transport) are defined
here. Every message:
  - Is @dataclass(frozen=True) (an immutable value type).
  - Has msg_type: MessageType as its first field (discriminant).
  - Has timestamp_utc: str (ISO 8601, millisecond precision) as its second field.
  - Has frame_id: int where applicable (uint32, monotonic counter).

Enums are imported from flight.libs.types.

Note: RawFrameMsg was removed in the raw-mosaic ingest contract change (spec Section 3).
Frames are passed by direct call from the sensor driver to the payload app (co-location
invariant); they never ride the bus. Use MosaicFrame from flight.libs.types instead.

Satisfies: REQ-AIML-COMP-001, REQ-AIML-COMP-002 (typed message-passing between processes).
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass
from datetime import datetime, timezone

# third-party
import numpy as np  # noqa: F401  (used in type comments)

# internal
from flight.libs.types import (
    AckStatus,
    DownlinkPriority,
    FaultCode,
    FrameUsabilityTag,
    GimbalCommandMode,
    GimbalState,
    LinkState,
    MessageType,
    SystemMode,
)

# ---------------------------------------------------------------------------
# Shared timestamp utility
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string with millisecond precision.

    Format: ``YYYY-MM-DDTHH:MM:SS.mmmZ``  (Z suffix, not +00:00, for compactness).

    All subsystems that construct messages with a ``timestamp_utc`` field must use
    this function to ensure a consistent format across the codebase. The Z suffix is
    expected by ``storage/writer.py``'s ``_make_frame_dir()`` parser.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"  # noqa: UP017


# ---------------------------------------------------------------------------
# Embedded sub-structs (not messages themselves, but embedded in messages)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlobMeta:
    """Metadata for a single detected plume blob.

    Embedded in InferenceResultMsg. blob_id is a uint16 persistent tracker ID
    assigned by controller/tracker.py and inherited across frames via IoU matching.
    """

    blob_id: int  # uint16 persistent tracker ID
    bbox: tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max) pixel space
    centroid_raw: tuple[float, float]  # (x, y) float centroid in crop-space pixels
    pixel_area: int  # number of pixels in blob mask
    mean_confidence: float  # mean softmax probability over blob pixels
    persistence_count: int  # consecutive frames this blob has been tracked


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessedFrameMsg:
    """Preprocessed, band-selected, calibrated tensor from preprocessing to inference.

    tensor shape: (4, H, W) float32, bands per InferenceConfig.input_bands
    (BLUE/GREEN/RED/NIR), H/W = sensor size / 2.
    """

    msg_type: MessageType  # must be MessageType.PROCESSED_FRAME
    timestamp_utc: str  # ISO 8601, millisecond precision
    frame_id: int  # uint32 monotonic frame counter
    tensor: object  # np.ndarray[float32, (4, H, W)]
    quality_flags: frozenset[FrameUsabilityTag]
    crop_origin_px: tuple[int, int]  # (x, y) top-left offset of crop in full frame
    scale_factor: float  # resize scale applied during preprocessing


@dataclass(frozen=True)
class InferenceResultMsg:
    """Segmentation output from inference to controller and storage.

    mask shape: (H, W) float32, softmax probability per pixel.
    """

    msg_type: MessageType  # must be MessageType.INFERENCE_RESULT
    timestamp_utc: str  # ISO 8601, millisecond precision
    frame_id: int  # uint32 monotonic frame counter
    mask: object  # np.ndarray[float32, (H, W)]
    blobs: tuple[BlobMeta, ...]  # zero or more detected blobs
    model_version: str  # model checkpoint identifier string
    inference_ms: float  # wall-clock inference duration in ms
    mode_flags: int  # uint8 bitmask; semantics defined in config
    crop_origin_px: tuple[int, int]  # (x, y) preprocess crop origin the blobs live in
    scale_factor: float  # preprocess decimation scale the blobs live in (tensor_px = plane_px * s)


@dataclass(frozen=True)
class GimbalCommandMsg:
    """Telemetry record of a gimbal command issued by the payload app.

    Reshaped from the legacy delta command into a typed telemetry record of the
    GimbalRequest the payload app issued onto the GimbalActuator HAL: mode plus the
    two axis values (interpreted per the mode), the arbiter state at the time, and a
    human-readable reason. This is a downlink/log record -- it is no longer the
    actuation vehicle (actuation flows through the HAL methods directly).
    """

    msg_type: MessageType  # must be MessageType.GIMBAL_COMMAND
    timestamp_utc: str  # ISO 8601, millisecond precision
    frame_id: int  # frame that triggered this command
    mode: GimbalCommandMode  # RATE / ABSOLUTE / STOW / HOME
    az_value_deg: float  # rate (deg/s) for RATE; target angle (deg) for ABSOLUTE; 0 otherwise
    el_value_deg: float  # rate (deg/s) for RATE; target angle (deg) for ABSOLUTE; 0 otherwise
    state: GimbalState  # arbiter state at time of command
    reason: str  # human-readable reason code for logging


@dataclass(frozen=True)
class TelemetryEventMsg:
    """Structured telemetry event from any subsystem to the telemetry reporter.

    payload must contain only JSON-serializable primitive types.
    """

    msg_type: MessageType  # must be MessageType.TELEMETRY_EVENT
    timestamp_utc: str  # ISO 8601, millisecond precision
    subsystem: str  # originating subsystem name (snake_case)
    event_name: str  # short snake_case event identifier
    payload: dict[str, str | int | float | bool]  # serializable structured fields only


@dataclass(frozen=True)
class FaultEventMsg:
    """Fault notification from any subsystem to the fault detection process."""

    msg_type: MessageType  # must be MessageType.FAULT_EVENT
    timestamp_utc: str  # ISO 8601, millisecond precision
    fault_code: FaultCode  # enumerated fault condition
    subsystem: str  # subsystem that detected/raised the fault
    detail: str  # human-readable fault detail string


@dataclass(frozen=True)
class HeartbeatMsg:
    """Periodic liveness signal from each subsystem to the fault watchdog."""

    msg_type: MessageType  # must be MessageType.HEARTBEAT
    timestamp_utc: str  # ISO 8601, millisecond precision
    subsystem: str  # originating subsystem name
    sequence: int  # monotonic heartbeat counter per subsystem


@dataclass(frozen=True)
class ModeChangeMsg:
    """System mode transition request or notification."""

    msg_type: MessageType  # must be MessageType.MODE_CHANGE
    timestamp_utc: str  # ISO 8601, millisecond precision
    new_mode: SystemMode  # requested target system mode
    requested_by: str  # subsystem or operator that requested the change


@dataclass(frozen=True)
class CommandMsg:
    """Ground/station command routed via iss_iface to a target subsystem.

    The standard command envelope: the station/ground sends a CommandMsg to iss_iface,
    which publishes it onto the bus for the core/target app to act on. params holds only
    JSON-serializable primitives. seq is a monotonic per-source counter for ordering and
    de-duplication.
    """

    msg_type: MessageType  # must be MessageType.COMMAND
    timestamp_utc: str  # ISO 8601, millisecond precision
    target: str  # destination subsystem name (e.g. "payload", "fault")
    command_id: str  # command identifier / opcode (e.g. "set_mode")
    params: dict[str, str | int | float | bool]  # serializable command parameters only
    source: str  # command origin (e.g. "ground", "station_ops")
    seq: int  # monotonic per-source command sequence number


@dataclass(frozen=True)
class RoutedCommandMsg:
    """A command the core router has accepted and dispatched to its target subsystem.

    Emitted by flight.core.command_router after a CommandMsg passes routing (known target,
    and for hazardous commands a valid ARM->EXECUTE two-step + inhibit pre-check). The target
    app consumes this (not the raw CommandMsg) and emits an execution CommandAckMsg. Carries
    the same envelope fields as the originating CommandMsg so the actuator can correlate the
    execution ack back to the ground command via (source, seq, command_id).
    """

    msg_type: MessageType  # must be MessageType.ROUTED_COMMAND
    timestamp_utc: str  # ISO 8601, millisecond precision
    target: str  # destination subsystem name (resolved from the command dictionary)
    command_id: str  # command identifier / opcode
    params: dict[str, str | int | float | bool]  # serializable command parameters only
    source: str  # command origin (echoed for ack correlation)
    seq: int  # per-source sequence number (echoed for ack correlation)


@dataclass(frozen=True)
class SafetyStateMsg:
    """Fault-owned safety state, published by the fault app each tick (inhibit authority).

    The single source of truth for SAFE-latch state and the active SAFE-triggering fault set.
    The command router subscribes to it to pre-check hazardous-command inhibits at routing
    time; the actuating apps still enforce their device interlocks at actuation (layered
    authority). active_faults is the set of SAFE-triggering fault codes observed in the most
    recent tick (empty once the triggering condition clears), which gates EXIT_SAFE.
    """

    msg_type: MessageType  # must be MessageType.SAFETY_STATE
    timestamp_utc: str  # ISO 8601, millisecond precision
    mode: SystemMode  # SAFE while latched, else IDLE
    active_faults: tuple[FaultCode, ...]  # SAFE-triggering faults seen this tick (sorted)
    safe_latched: bool  # True once a SAFE-triggering fault latched SAFE, until EXIT_SAFE
    safe_reason: FaultCode  # the fault that latched SAFE (NONE when not latched)


@dataclass(frozen=True)
class StorageWriteMsg:
    """Bundle of a full frame's data for the storage writer process.

    raw_frame shape: (C, H, W) float32.
    processed_tensor shape: (4, H, W) float32.
    """

    msg_type: MessageType  # must be MessageType.STORAGE_WRITE
    timestamp_utc: str  # ISO 8601, millisecond precision
    frame_id: int  # uint32 monotonic frame counter
    raw_frame: object  # np.ndarray[float32, (C, H, W)]
    processed_tensor: object  # np.ndarray[float32, (4, H, W)]
    inference_result: InferenceResultMsg  # full inference output for this frame
    usability: FrameUsabilityTag  # computed usability classification


@dataclass(frozen=True)
class ProductRefMsg:
    """Compact reference to a stored science product, published after the payload stores it.

    The payload persists large artifacts (mask thumbnails) via the injected StorageWriter
    (bypassing the bus, per the large-artifact invariant) and publishes only this compact
    reference. The downlink manager enqueues it by priority; iss_iface fetches the bytes from
    storage via the injected StorageReader at transmission time. Keeps tensors/masks off the bus.
    """

    msg_type: MessageType  # must be MessageType.PRODUCT_REF
    timestamp_utc: str  # ISO 8601, millisecond precision
    entry_id: str  # storage entry id returned by StorageWriter.store
    priority: DownlinkPriority  # downlink priority (typically SCIENCE_PRODUCT)
    item_id: str  # human-readable product identifier (e.g. "mask_thumb_<frame>")
    byte_len: int  # size of the stored product in bytes (for budget accounting)


@dataclass(frozen=True)
class DownlinkItemMsg:
    """A single item the downlink manager has selected for CCSDS downlink, in priority order.

    Produced only by the downlink manager (the sole prioritizer) and consumed only by
    iss_iface (the link egress). Carries either inline payload_bytes (compact items: faults,
    acks, HK telemetry) or a non-empty storage_ref naming a stored product iss_iface fetches
    via the injected StorageReader at transmission time (large-artifact invariant).
    """

    msg_type: MessageType  # must be MessageType.DOWNLINK_ITEM
    timestamp_utc: str  # ISO 8601, millisecond precision
    priority: DownlinkPriority  # queue priority (lower int == higher priority)
    payload_bytes: bytes  # serialized inline content (empty when storage_ref is set)
    crc32: int  # CRC-32 of payload_bytes (0 for storage_ref items)
    item_id: str  # unique item identifier string
    storage_ref: str = ""  # storage entry id to fetch at tx time; "" => inline payload_bytes


@dataclass(frozen=True)
class UploadChunkMsg:
    """One chunk of a chunked model upload via CCSDS uplink.

    Required by comms/uplink.py. Added per section 8 known-gap note in types/CLAUDE.md.
    chunk_index is zero-based. data contains the raw bytes for this chunk.
    expected_crc32 is the CRC-32 of the complete reassembled model file (not this chunk).
    """

    msg_type: MessageType  # must be MessageType.UPLINK_CHUNK
    timestamp_utc: str  # ISO 8601, millisecond precision
    chunk_index: int  # zero-based chunk index
    total_chunks: int  # total number of chunks in this upload session
    data: bytes  # raw bytes for this chunk
    expected_crc32: int  # CRC-32 of the complete reassembled file


@dataclass(frozen=True)
class CommandAckMsg:
    """Acknowledgement (positive or negative) for one inbound ground command.

    Emitted by iss_iface for every inbound packet (ingress accept/reject) and by target
    apps/services on execution (Phase 6B). Correlates back to the originating command via
    (source, seq, command_id). On REJECTED, fault_code carries the reason; on ACCEPTED it
    is FaultCode.NONE.
    """

    msg_type: MessageType  # must be MessageType.COMMAND_ACK
    timestamp_utc: str  # ISO 8601, millisecond precision
    status: AckStatus  # ACCEPTED or REJECTED
    command_id: str  # echoed opcode string ("" if the body was unparseable)
    source: str  # echoed command origin
    seq: int  # echoed per-source sequence number (-1 if unparseable)
    fault_code: FaultCode  # NONE on ACCEPTED; the reject reason otherwise
    detail: str  # human-readable reason / context


@dataclass(frozen=True)
class LinkStateMsg:
    """Current station-link acquisition state, published by iss_iface each tick."""

    msg_type: MessageType  # must be MessageType.LINK_STATE
    timestamp_utc: str  # ISO 8601, millisecond precision
    state: LinkState  # AOS (link up) or LOS (link down)
