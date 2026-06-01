"""HAL device interfaces (Protocols). Apps depend only on this module, never on
concrete drivers; the composition root injects the implementation.
"""

from flight.hal.interfaces.gimbal import GimbalActuator, GimbalPosition
from flight.hal.interfaces.sensor import ImagingSensor

__all__ = ["GimbalActuator", "GimbalPosition", "ImagingSensor"]
