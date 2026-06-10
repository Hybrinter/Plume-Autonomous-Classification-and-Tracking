"""Pure flight types: enumerations, the Result[T, E] error type, and raw-frame types.

Other flight modules import these from `flight.libs.types`, never from the
submodules, so the internal split stays refactorable.

Exports:
- Enumerations: Band, DownlinkPriority, FaultCode, FrameUsabilityTag, GimbalState,
  MessageType, ModelDeployState, SystemMode.
- Result types: Err, Ok, Result.
- Frame types: MosaicFrame.
"""

from flight.libs.types.enums import (
    Band,
    DownlinkPriority,
    FaultCode,
    FrameUsabilityTag,
    GimbalState,
    MessageType,
    ModelDeployState,
    SystemMode,
)
from flight.libs.types.frames import MosaicFrame
from flight.libs.types.result import Err, Ok, Result

__all__ = [
    "Band",
    "DownlinkPriority",
    "Err",
    "FaultCode",
    "FrameUsabilityTag",
    "GimbalState",
    "MessageType",
    "MosaicFrame",
    "ModelDeployState",
    "Ok",
    "Result",
    "SystemMode",
]
