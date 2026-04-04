"""PACT types package public API.

Re-exports all public names from enums, messages, and config submodules.
All other PACT subsystems import types exclusively from this package — never
from the submodules directly — to keep the internal structure refactorable.

Import order: enums first (no deps), then messages (depends on enums),
then config (no deps on messages or enums at runtime).
"""

# stdlib
# (none)

# Re-export enums
from pact.types.enums import (
    DownlinkPriority,
    Err,
    FaultCode,
    FrameUsabilityTag,
    GimbalState,
    MessageType,
    ModelDeployState,
    Ok,
    Result,
    SystemMode,
)

# Re-export messages
from pact.types.messages import (
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
)

# Re-export config dataclasses
from pact.types.config import (
    CommsConfig,
    ControllerConfig,
    FaultConfig,
    InferenceConfig,
    PactConfig,
    StorageConfig,
)

__all__ = [
    # enums
    "SystemMode",
    "GimbalState",
    "FaultCode",
    "FrameUsabilityTag",
    "MessageType",
    "DownlinkPriority",
    "ModelDeployState",
    # Result types
    "Result",
    "Ok",
    "Err",
    # messages
    "BlobMeta",
    "RawFrameMsg",
    "ProcessedFrameMsg",
    "InferenceResultMsg",
    "GimbalCommandMsg",
    "TelemetryEventMsg",
    "FaultEventMsg",
    "HeartbeatMsg",
    "ModeChangeMsg",
    "StorageWriteMsg",
    "DownlinkItemMsg",
    "UploadChunkMsg",
    # config
    "ControllerConfig",
    "InferenceConfig",
    "CommsConfig",
    "StorageConfig",
    "FaultConfig",
    "PactConfig",
]
