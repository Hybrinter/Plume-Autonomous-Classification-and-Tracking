"""Flight composition root: construct real drivers and run the subsystem scheduler.

This is the production entry on the payload computer. It loads config, constructs the
real HAL drivers and the ONNX detector, wires every app via build_apps, and runs them
under the thread Scheduler until interrupted. Real drivers and onnxruntime are present
only on flight hardware, so this module is constructed/run at runtime, not in CI; the
driver-agnostic wiring it relies on (build_apps) is unit-tested with sim drivers.

Contains:
  - build_flight_system: construct the real Drivers bundle and the SystemApps.
  - main: load config, build the system, and run the scheduler until interrupted.
"""

from __future__ import annotations

# stdlib
import threading

# internal
from flight.core.composition import MONITORED_SUBSYSTEMS, Drivers, SystemApps, build_apps
from flight.core.config_loader import load_config
from flight.core.scheduler import Scheduler
from flight.hal.drivers_real import RealGimbal, RealScalarSensor, RealSensor, RealStationLink
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.time import Clock, RealClock
from flight.libs.types import Ok
from flight.payload.model import OnnxDetector


def build_flight_system(config: PactConfig, bus: MessageBus, clock: Clock) -> SystemApps:
    """Construct the real-driver Drivers bundle and wire the SystemApps.

    Args:
        config: The validated PactConfig.
        bus: The shared MessageBus.
        clock: The injected Clock (RealClock in production).

    Returns:
        The wired SystemApps.

    Notes:
        RealSensor lazily imports PySpin and OnnxDetector lazily imports onnxruntime;
        both raise ImportError if the SDK is absent. This function therefore runs only
        on flight hardware.
    """
    drivers = Drivers(
        sensor=RealSensor(),
        gimbal=RealGimbal(),
        detector=OnnxDetector(config.inference.model_path),
        station=RealStationLink(),
        thermal_sensor=RealScalarSensor(),
        power_sensor=RealScalarSensor(),
    )
    return build_apps(config, bus, clock, drivers, MONITORED_SUBSYSTEMS)


def main(config_path: str = "config/default.toml") -> None:
    """Load config, build the flight system, and run the scheduler until interrupted.

    Args:
        config_path: Path to the TOML config file.

    Raises:
        SystemExit: If config loading fails (unrecoverable startup error).
    """
    result = load_config(config_path)
    if not isinstance(result, Ok):
        raise SystemExit(f"config load failed: {result.error}")
    config = result.value

    bus = MessageBus()
    clock: Clock = RealClock()
    apps = build_flight_system(config, bus, clock)

    scheduler = Scheduler(
        [
            ("payload", apps.payload),
            ("fault", apps.fault),
            ("iss_iface", apps.iss_iface),
            ("thermal", apps.thermal),
            ("electrical", apps.electrical),
        ]
    )
    scheduler.start()
    try:
        threading.Event().wait()  # run until the process is signaled/interrupted
    except KeyboardInterrupt:
        scheduler.stop()
