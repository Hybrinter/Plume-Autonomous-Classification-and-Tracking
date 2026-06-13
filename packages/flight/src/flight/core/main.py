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
from flight.payload.calibration_io import build_identity_calibration, load_calibration
from flight.payload.model import OnnxDetector
from flight.payload.preprocess import MosaicCalibration


def build_flight_system(
    config: PactConfig, bus: MessageBus, clock: Clock, calib: MosaicCalibration
) -> SystemApps:
    """Construct the real-driver Drivers bundle and wire the SystemApps.

    Args:
        config: The validated PactConfig.
        bus: The shared MessageBus.
        clock: The injected Clock (RealClock in production).
        calib: The MosaicCalibration to inject into the payload app (loaded from
            checksummed artifacts, or identity when no calibration_dir is configured).

    Returns:
        The wired SystemApps.

    Raises:
        SystemExit: If the startup exposure/gain tuning fails (camera unusable at
            startup is unrecoverable).
        ValueError: If config.gimbal.serial_port is empty (RealGimbal cannot open its
            link; a misconfigured gimbal port is an unrecoverable startup failure).

    Notes:
        RealSensor lazily imports PySpin, RealGimbal lazily imports pyserial, and
        OnnxDetector lazily imports onnxruntime; each raises ImportError if its SDK is
        absent. This function therefore runs only on flight hardware. The startup
        exposure/gain are commanded from config.sensor before the apps are wired.
        RealStationLink binds its TCP server socket in __init__; ValueError is raised if
        config.link contains an empty host or an out-of-range port (startup misconfig).
    """
    sensor = RealSensor(clock=clock)
    exposure_result = sensor.set_exposure_us(config.sensor.default_exposure_us)
    if not isinstance(exposure_result, Ok):
        raise SystemExit(f"camera exposure setup failed: {exposure_result.error}")
    gain_result = sensor.set_gain_db(config.sensor.default_gain_db)
    if not isinstance(gain_result, Ok):
        raise SystemExit(f"camera gain setup failed: {gain_result.error}")
    drivers = Drivers(
        sensor=sensor,
        gimbal=RealGimbal(clock=clock, cfg=config.gimbal),
        detector=OnnxDetector(config.inference.model_path),
        station=RealStationLink(cfg=config.link, clock=clock),
        thermal_sensor=RealScalarSensor(),
        power_sensor=RealScalarSensor(),
    )
    return build_apps(config, bus, clock, drivers, MONITORED_SUBSYSTEMS, calib)


def main(config_path: str = "config/default.toml") -> None:
    """Load config, build the flight system, and run the scheduler until interrupted.

    Args:
        config_path: Path to the TOML config file.

    Raises:
        SystemExit: If config loading or calibration loading fails (unrecoverable
            startup errors).
    """
    result = load_config(config_path)
    if not isinstance(result, Ok):
        raise SystemExit(f"config load failed: {result.error}")
    config = result.value

    if config.sensor.calibration_dir:
        cal_result = load_calibration(
            config.sensor.calibration_dir, config.sensor.height_px, config.sensor.width_px
        )
        if not isinstance(cal_result, Ok):
            raise SystemExit(f"calibration load failed: {cal_result.error}")
        calib = cal_result.value
    else:
        calib = build_identity_calibration(config.sensor.height_px, config.sensor.width_px)

    bus = MessageBus()
    clock: Clock = RealClock()
    apps = build_flight_system(config, bus, clock, calib)

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
