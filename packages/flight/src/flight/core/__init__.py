"""Compute / C&DH host: config loading, the composition root, and scheduling."""

from flight.core.composition import MONITORED_SUBSYSTEMS, Drivers, SystemApps, build_apps
from flight.core.config_loader import load_config
from flight.core.main import build_flight_system, main
from flight.core.scheduler import RunnableApp, Scheduler

__all__ = [
    "MONITORED_SUBSYSTEMS",
    "Drivers",
    "RunnableApp",
    "Scheduler",
    "SystemApps",
    "build_apps",
    "build_flight_system",
    "load_config",
    "main",
]
