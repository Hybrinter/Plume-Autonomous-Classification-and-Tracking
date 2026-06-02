"""HAL device interfaces (Protocols). Apps depend only on this module, never on
concrete drivers; the composition root injects the implementation.
"""

from flight.hal.interfaces.gimbal import GimbalActuator, GimbalPosition
from flight.hal.interfaces.scalar import ScalarSensor
from flight.hal.interfaces.sensor import ImagingSensor
from flight.hal.interfaces.station import StationLink

__all__ = ["GimbalActuator", "GimbalPosition", "ImagingSensor", "ScalarSensor", "StationLink"]
