"""HAL device interfaces (Protocols). Apps depend only on this module, never on
concrete drivers; the composition root injects the implementation.
"""

from flight.hal.interfaces.gimbal import GimbalActuator, GimbalPosition
from flight.hal.interfaces.launch_lock import LaunchLock
from flight.hal.interfaces.scalar import ScalarSensor
from flight.hal.interfaces.sensor import ImagingSensor
from flight.hal.interfaces.station import StationLink
from flight.hal.interfaces.storage import StorageReader, StorageWriter

__all__ = [
    "GimbalActuator",
    "GimbalPosition",
    "ImagingSensor",
    "LaunchLock",
    "ScalarSensor",
    "StationLink",
    "StorageReader",
    "StorageWriter",
]
