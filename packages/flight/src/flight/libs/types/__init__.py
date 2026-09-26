"""Pure flight types: enumerations, the Result[T, E] error type, and raw-frame types.

Other flight modules import these from `flight.libs.types`, never from the
submodules, so the internal split stays refactorable.

Exports:
- Enumerations: AckStatus, Band, CommandId, DownlinkPriority, FaultCode, FrameUsabilityTag,
  GimbalCommandMode, GimbalState, LinkState, MessageType, ModelDeployState, ParamKind,
  SystemMode.
- Result types: Err, Ok, Result.
- Frame types: MosaicFrame.
"""

from flight.libs.types.enums import (
    AckStatus,
    Band,
    CommandId,
    DownlinkPriority,
    FaultCode,
    FrameUsabilityTag,
    GimbalCommandMode,
    GimbalState,
    LinkState,
    MessageType,
    ModelDeployState,
    ParamKind,
    SystemMode,
)
from flight.libs.types.frames import MosaicFrame
from flight.libs.types.result import Err, Ok, Result

__all__ = [
    "AckStatus",
    "Band",
    "CommandId",
    "DownlinkPriority",
    "Err",
    "FaultCode",
    "FrameUsabilityTag",
    "GimbalCommandMode",
    "GimbalState",
    "LinkState",
    "MessageType",
    "ModelDeployState",
    "MosaicFrame",
    "Ok",
    "ParamKind",
    "Result",
    "SystemMode",
]
