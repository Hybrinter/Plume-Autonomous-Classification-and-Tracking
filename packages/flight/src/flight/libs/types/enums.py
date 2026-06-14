"""Flight enumeration types.

Defines all enumerations used as discriminants and state values across the flight
software. The Ok/Err/Result types live in flight.libs.types.result.

Includes:
- SystemMode: top-level operational mode transitions.
- GimbalState: four-state arbiter for gimbal control.
- GimbalCommandMode: interpretation of gimbal command axis values (RATE/ABSOLUTE/STOW/HOME).
- FaultCode: all enumerated fault conditions, including ingest-chain codes
  (CALIBRATION_INVALID, FRAME_MALFORMED), driver-level gimbal fault (GIMBAL_FAULT), and
  command-ingress integrity codes (COMMAND_CRC_FAIL, COMMAND_AUTH_FAIL, COMMAND_SEQ_ERROR,
  COMMAND_INVALID).
- Band: physical 2x2 mosaic-filter band vocabulary (BLUE/GREEN/RED/NIR).
- FrameUsabilityTag: per-frame quality classification.
- MessageType: typed discriminant for all bus messages.
- DownlinkPriority: downlink queue priority.
- ModelDeployState: model deployment lifecycle state.
- LinkState: station link acquisition state (AOS/LOS).
- AckStatus: outcome of an inbound command at ingress (ACCEPTED/REJECTED).
- CommandId: opcode keys for the typed command dictionary.
- ParamKind: primitive kind for command parameter schema validation.

Satisfies: REQ-AIML-COMP-001, REQ-AIML-COMP-002 (type-safety foundation for all subsystems).

No other flight module is imported here. This module is a dependency root.
"""

from __future__ import annotations

# stdlib
import enum

# ---------------------------------------------------------------------------
# System-level enumerations
# ---------------------------------------------------------------------------


class SystemMode(enum.Enum):
    """Top-level operational mode. REQ-OPER-HIGH-002."""

    IDLE = "IDLE"
    ACTIVE = "ACTIVE"  # inference + gimbal running
    SCAN = "SCAN"  # nadir scan, no active target
    MODEL_UPLINK = "MODEL_UPLINK"
    DATA_DOWNLINK = "DATA_DOWNLINK"
    SAFE = "SAFE"  # fault-induced; minimal activity


class GimbalState(enum.Enum):
    """Four-state + safe arbiter. REQ-AIML-GIMB-008."""

    IDLE = "IDLE"
    ACQUIRING = "ACQUIRING"
    TRACKING = "TRACKING"
    SCAN = "SCAN"
    SAFE = "SAFE"


class GimbalCommandMode(enum.Enum):
    """How a gimbal command's axis values are interpreted.

    RATE: az/el are rates in deg/s (TRACKING). ABSOLUTE: az/el are target angles in
    degrees (SCAN, acquisition repositioning). STOW/HOME: axis values are ignored;
    the driver resolves the configured stow/home pose.

    String values mirror member names (log readability convention).
    Satisfies: REQ-AIML-GIMB-001, REQ-GIMB-HIGH-001.
    """

    RATE = "RATE"
    ABSOLUTE = "ABSOLUTE"
    STOW = "STOW"
    HOME = "HOME"


class FaultCode(enum.Enum):
    """Enumerated fault conditions."""

    NONE = "NONE"
    INFERENCE_TIMEOUT = "INFERENCE_TIMEOUT"
    INFERENCE_NAN = "INFERENCE_NAN"
    CAMERA_STALL = "CAMERA_STALL"
    STORAGE_FULL = "STORAGE_FULL"
    THERMAL_OVER_LIMIT = "THERMAL_OVER_LIMIT"
    POWER_OVER_LIMIT = "POWER_OVER_LIMIT"
    GIMBAL_RUNAWAY = "GIMBAL_RUNAWAY"
    COMM_TIMEOUT = "COMM_TIMEOUT"
    WATCHDOG_EXPIRE = "WATCHDOG_EXPIRE"
    MODEL_CORRUPT = "MODEL_CORRUPT"
    PROCESS_DIED = "PROCESS_DIED"
    CALIBRATION_INVALID = "CALIBRATION_INVALID"
    FRAME_MALFORMED = "FRAME_MALFORMED"
    GIMBAL_FAULT = "GIMBAL_FAULT"
    COMMAND_CRC_FAIL = "COMMAND_CRC_FAIL"
    COMMAND_AUTH_FAIL = "COMMAND_AUTH_FAIL"
    COMMAND_SEQ_ERROR = "COMMAND_SEQ_ERROR"
    COMMAND_INVALID = "COMMAND_INVALID"


class Band(enum.Enum):
    """Physical 2x2 mosaic-filter band names.

    Passbands approximate Sentinel-2: BLUE ~490 nm (B2), GREEN ~560 nm (B3),
    RED ~665 nm (B4), NIR ~842 nm (B8) -- chosen so Sentinel-2-derived training
    data remains a valid domain (spec Section 2).

    String values mirror member names (log readability convention).
    """

    BLUE = "BLUE"
    GREEN = "GREEN"
    RED = "RED"
    NIR = "NIR"


class FrameUsabilityTag(enum.Enum):
    """Per-frame usability classification. REQ-AIML-DATA-005."""

    TRAINING = "TRAINING"
    TRACKING = "TRACKING"
    INVALID = "INVALID"
    CLOUD_CONTAMINATED = "CLOUD_CONTAMINATED"
    SUNGLINT = "SUNGLINT"
    SATURATED = "SATURATED"
    MOTION_SMEAR = "MOTION_SMEAR"
    INCOMPLETE_METADATA = "INCOMPLETE_METADATA"


class MessageType(enum.Enum):
    """Discriminant for all inter-process messages."""

    PROCESSED_FRAME = "PROCESSED_FRAME"
    INFERENCE_RESULT = "INFERENCE_RESULT"
    GIMBAL_COMMAND = "GIMBAL_COMMAND"
    TELEMETRY_EVENT = "TELEMETRY_EVENT"
    FAULT_EVENT = "FAULT_EVENT"
    HEARTBEAT = "HEARTBEAT"
    MODE_CHANGE = "MODE_CHANGE"
    COMMAND = "COMMAND"
    STORAGE_WRITE = "STORAGE_WRITE"
    DOWNLINK_ITEM = "DOWNLINK_ITEM"
    UPLINK_CHUNK = "UPLINK_CHUNK"
    COMMAND_ACK = "COMMAND_ACK"
    LINK_STATE = "LINK_STATE"


class DownlinkPriority(enum.Enum):
    """Downlink queue priority. REQ-COMM-HIGH-001.

    Lower integer value == higher priority (used directly by queue.PriorityQueue).
    """

    HEALTH_TELEMETRY = 0  # highest priority
    SCIENCE_DATA = 1
    COMPRESSED_IMAGERY = 2
    RAW_IMAGERY = 3  # lowest priority


class ModelDeployState(enum.Enum):
    """Model deployment lifecycle state. REQ-AIML-HIGH-004."""

    ACTIVE = "ACTIVE"
    STAGED = "STAGED"
    ROLLBACK_AVAILABLE = "ROLLBACK_AVAILABLE"


class LinkState(enum.Enum):
    """Station link acquisition state. AOS = link up (drain downlink), LOS = link down.

    String values mirror member names (log readability convention). Satisfies: REQ-COMM-HIGH-001.
    """

    AOS = "AOS"  # acquisition of signal: contact established, downlink may drain
    LOS = "LOS"  # loss of signal: no contact, hold downlink


class AckStatus(enum.Enum):
    """Outcome of a single inbound command at ingress.

    String values mirror member names (log readability convention). Satisfies: REQ-COMM-HIGH-004.
    """

    ACCEPTED = "ACCEPTED"  # decoded, authenticated, and validated; CommandMsg published
    REJECTED = "REJECTED"  # failed CRC / auth / sequence / dictionary validation; no CommandMsg


class CommandId(enum.Enum):
    """The command dictionary's opcode keys (per-command schema lives in flight.libs.commands).

    String values mirror member names (log readability convention). Satisfies: REQ-COMM-HIGH-003.
    """

    PING = "PING"  # liveness check; non-hazardous; no params
    SET_THERMAL_LIMIT = "SET_THERMAL_LIMIT"  # non-hazardous; param limit_c: float
    NOOP = "NOOP"  # accepted no-op; non-hazardous; no params


class ParamKind(enum.Enum):
    """Primitive kind a command parameter must be, for dictionary validation.

    String values mirror member names (log readability convention). Satisfies: REQ-COMM-HIGH-003.
    """

    STR = "STR"
    INT = "INT"
    FLOAT = "FLOAT"
    BOOL = "BOOL"
