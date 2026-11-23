"""Pure flight types: enumerations, the Result[T, E] error type, and raw-frame types.

Other flight modules import these from `flight.libs.types`, never from the
submodules, so the internal split stays refactorable.

Exports:
- Enumerations: AckStatus, Band, CommandId, DownlinkPriority, FaultCode,
  FrameUsabilityTag, GimbalCommandMode, LinkState, MessageType,
  ModeTransitionDecision, ModelDeployState, ParamKind, SystemMode.
- Activation types: ActivationKey.
- Result types: Err, Ok, Result.
- Frame types: MosaicFrame.
"""

from flight.libs.types.activation import ActivationKey
from flight.libs.types.enums import (
    AckStatus,
    Band,
    CommandId,
    DownlinkPriority,
    FaultCode,
    FrameUsabilityTag,
    GimbalCommandMode,
    LinkState,
    MessageType,
    ModelDeployState,
    ModeTransitionDecision,
    ParamKind,
    SystemMode,
)
from flight.libs.types.frames import MosaicFrame
from flight.libs.types.result import Err, Ok, Result

__all__ = [
    "AckStatus",
    "ActivationKey",
    "Band",
    "CommandId",
    "DownlinkPriority",
    "Err",
    "FaultCode",
    "FrameUsabilityTag",
    "GimbalCommandMode",
    "LinkState",
    "MessageType",
    "ModeTransitionDecision",
    "ModelDeployState",
    "MosaicFrame",
    "Ok",
    "ParamKind",
    "Result",
    "SystemMode",
]
