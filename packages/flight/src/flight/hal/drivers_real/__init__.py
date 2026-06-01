"""Flight hardware HAL drivers. Reachable only from composition roots (flight/core).

Importing this module is safe without any hardware SDK; constructing RealSensor
lazily imports PySpin and raises ImportError if it is absent.
"""

from flight.hal.drivers_real.gimbal import RealGimbal
from flight.hal.drivers_real.sensor import RealSensor

__all__ = ["RealGimbal", "RealSensor"]
