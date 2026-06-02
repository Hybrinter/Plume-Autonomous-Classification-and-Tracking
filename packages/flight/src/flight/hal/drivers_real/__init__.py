"""Flight hardware HAL drivers. Reachable only from composition roots (flight/core).

Importing this module is safe without any hardware SDK; constructing RealSensor
lazily imports PySpin and raises ImportError if it is absent.
"""

from flight.hal.drivers_real.gimbal import RealGimbal
from flight.hal.drivers_real.scalar import RealScalarSensor
from flight.hal.drivers_real.sensor import RealSensor
from flight.hal.drivers_real.station import RealStationLink

__all__ = ["RealGimbal", "RealScalarSensor", "RealSensor", "RealStationLink"]
