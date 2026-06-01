"""Simulation HAL drivers. Reachable only from composition roots (sim/SIL)."""

from flight.hal.drivers_sim.gimbal import SimGimbal
from flight.hal.drivers_sim.sensor import SimSensor

__all__ = ["SimGimbal", "SimSensor"]
