"""Flight enumeration types.

Defines all enumerations used as discriminants and state values across the flight
software. The Ok/Err/Result types live in flight.libs.types.result.

Includes:
- SystemMode: top-level operational mode transitions.
- GimbalState: TRACKING / REWIND / SAFE arbiter for gimbal control.
- GimbalCommandMode: interpretation of gimbal pose commands (ABSOLUTE/STOW/HOME).
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
    """Three-state + safe arbiter. REQ-AIML-GIMB-008."""

    TRACKING = "TRACKING"
    REWIND = "REWIND"
    SAFE = "SAFE"


class GimbalCommandMode(enum.Enum):
    """How a gimbal pose command's elevation is interpreted.

    ABSOLUTE: el is a target angle in degrees (HOME/GOTO via the position loop).
    STOW/HOME: elevation is ignored; the driver or position loop uses the configured pose.

    String values mirror member names (log readability convention).
    Satisfies: REQ-AIML-GIMB-001, REQ-GIMB-HIGH-001.
    """

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
    STORAGE_CORRUPT = "STORAGE_CORRUPT"
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
    EPHEMERIS_FAULT = "EPHEMERIS_FAULT"
    COMMAND_CRC_FAIL = "COMMAND_CRC_FAIL"
    COMMAND_AUTH_FAIL = "COMMAND_AUTH_FAIL"
    COMMAND_SEQ_ERROR = "COMMAND_SEQ_ERROR"
    COMMAND_INVALID = "COMMAND_INVALID"
    COMMAND_UNROUTABLE = "COMMAND_UNROUTABLE"
    LAUNCH_LOCK_FAULT = "LAUNCH_LOCK_FAULT"


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
    ROUTED_COMMAND = "ROUTED_COMMAND"
    SAFETY_STATE = "SAFETY_STATE"
    STORAGE_WRITE = "STORAGE_WRITE"
    PRODUCT_REF = "PRODUCT_REF"
    DOWNLINK_ITEM = "DOWNLINK_ITEM"
    UPLINK_CHUNK = "UPLINK_CHUNK"
    COMMAND_ACK = "COMMAND_ACK"
    LINK_STATE = "LINK_STATE"
    LAUNCH_LOCK_STATE = "LAUNCH_LOCK_STATE"
    MODEL_STAGED = "MODEL_STAGED"
    MODEL_DEPLOY = "MODEL_DEPLOY"


class DownlinkPriority(enum.Enum):
    """Downlink queue priority. REQ-COMM-HIGH-001.

    Lower integer value == higher priority (used directly by the downlink manager's ordering).
    The order encodes spec Section 6: fault events > command acks > housekeeping telemetry >
    science products.
    """

    FAULT_EVENT = 0  # highest priority
    COMMAND_ACK = 1
    HK_TELEMETRY = 2
    SCIENCE_PRODUCT = 3  # lowest priority


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


class LaunchLockState(enum.Enum):
    """Launch-lock mechanism state (motorized pin with engaged/released microswitches).

    String values mirror member names (log readability convention). The lock starts ENGAGED
    (flight configuration); release is a hazardous ground-commanded operation. UNKNOWN is
    reported when the microswitches disagree or a read fails. Satisfies: REQ-MECH-HIGH-001.
    """

    ENGAGED = "ENGAGED"  # pin engaged: gimbal motion inhibited
    RELEASED = "RELEASED"  # pin released: gimbal free to move
    UNKNOWN = "UNKNOWN"  # indeterminate (switch disagreement / read failure)


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

    PING = "PING"  # liveness check; non-hazardous; core-handled; no params
    SET_THERMAL_LIMIT = "SET_THERMAL_LIMIT"  # non-hazardous; target thermal; param limit_c: float
    NOOP = "NOOP"  # accepted no-op; non-hazardous; core-handled; no params
    EXIT_SAFE = "EXIT_SAFE"  # hazardous (ARM/EXECUTE); target fault; param phase: str
    RELEASE_LAUNCH_LOCK = "RELEASE_LAUNCH_LOCK"  # hazardous; target mechanical; param phase: str
    UPLOAD_MODEL_CHUNK = "UPLOAD_MODEL_CHUNK"  # non-hazardous; target iss_iface; chunked uplink
    ACTIVATE_MODEL = "ACTIVATE_MODEL"  # non-hazardous; target model_deploy; activate staged model


class ParamKind(enum.Enum):
    """Primitive kind a command parameter must be, for dictionary validation.

    String values mirror member names (log readability convention). Satisfies: REQ-COMM-HIGH-003.
    """

    STR = "STR"
    INT = "INT"
    FLOAT = "FLOAT"
    BOOL = "BOOL"
