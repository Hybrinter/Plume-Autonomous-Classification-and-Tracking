"""Pure flight types: enumerations and the Result[T, E] error type.

Other flight modules import these from `flight.libs.types`, never from the
submodules, so the internal split stays refactorable.
"""

from flight.libs.types.enums import (
    DownlinkPriority,
    FaultCode,
    FrameUsabilityTag,
    GimbalState,
    MessageType,
    ModelDeployState,
    SystemMode,
)
from flight.libs.types.result import Err, Ok, Result

__all__ = [
    "DownlinkPriority",
    "Err",
    "FaultCode",
    "FrameUsabilityTag",
    "GimbalState",
    "MessageType",
    "ModelDeployState",
    "Ok",
    "Result",
    "SystemMode",
]
