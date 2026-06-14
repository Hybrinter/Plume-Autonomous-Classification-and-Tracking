"""Flight composition root: construct real drivers and run the subsystem scheduler.

This is the production entry on the payload computer. It loads config, constructs the
real HAL drivers and the ONNX detector, wires every app via build_apps, and runs them
under the thread Scheduler until interrupted. Real drivers and onnxruntime are present
only on flight hardware, so this module is constructed/run at runtime, not in CI; the
driver-agnostic wiring it relies on (build_apps) is unit-tested with sim drivers.

Contains:
  - build_flight_system: resolve the env-selected Drivers bundle and wire the SystemApps.
  - main: load config, build the system, and run the scheduler until interrupted.
"""

from __future__ import annotations

# stdlib
import threading

# internal
from flight.core.composition import MONITORED_SUBSYSTEMS, SystemApps, build_apps
from flight.core.config_loader import load_config
from flight.core.scheduler import Scheduler
from flight.core.select_drivers import select_drivers
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.time import Clock, ManualClock, RealClock
from flight.libs.types import Ok
from flight.payload.calibration_io import build_identity_calibration, load_calibration
from flight.payload.preprocess import MosaicCalibration


def _load_uplink_key(path: str) -> bytes:
    """Load the shared HMAC-SHA256 uplink secret from a binary file.

    Args:
        path: Filesystem path to the key file (raw bytes, no encoding).

    Returns:
        The key bytes.

    Raises:
        SystemExit: If the file does not exist or cannot be read (a missing uplink key is
            an unrecoverable startup misconfig; the vehicle must not accept unauthenticated
            commands in flight).
    """
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError as exc:
        raise SystemExit(f"uplink key load failed ({path}): {exc}") from exc


def build_flight_system(
    config: PactConfig, bus: MessageBus, clock: Clock, calib: MosaicCalibration
) -> SystemApps:
    """Resolve the env-selected Drivers bundle and wire the SystemApps.

    Args:
        config: The validated PactConfig (its environment axes select each driver).
        bus: The shared MessageBus.
        clock: The injected Clock (chosen in main from config.environment.clock).
        calib: The MosaicCalibration to inject into the payload app (loaded from
            checksummed artifacts, or identity when no calibration_dir is configured).

    Returns:
        The wired SystemApps.

    Raises:
        SystemExit: If the uplink key file is missing/unreadable, or if the
            real-sensor startup exposure/gain tuning fails (both unrecoverable at
            startup; the latter now lives inside select_drivers).
        ValueError: If a 'real' gimbal is selected with an empty config.gimbal.serial_port
            (RealGimbal cannot open its link -- an unrecoverable startup misconfig).

    Notes:
        Driver construction is delegated to flight.core.select_drivers, which lazily
        imports PySpin/pyserial/onnxruntime only inside the 'real' branches it backs.
        With the default all-"real" environment this builds the full hardware stack, so
        this function runs only on flight hardware. sim_inputs is None: the default flight
        env has no 'sim' axis, so no sim construction inputs are needed (select_drivers
        raises ValueError if that assumption is ever violated by a misconfigured env).
    """
    uplink_key = _load_uplink_key(config.command_ingress.hmac_key_path)
    drivers = select_drivers(config, clock, sim_inputs=None)
    return build_apps(config, bus, clock, drivers, MONITORED_SUBSYSTEMS, calib, uplink_key)


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
    clock: Clock = RealClock() if config.environment.clock == "real" else ManualClock()
    apps = build_flight_system(config, bus, clock, calib)

    scheduler = Scheduler(
        [
            ("payload", apps.payload),
            ("fault", apps.fault),
            ("iss_iface", apps.iss_iface),
            ("thermal", apps.thermal),
            ("electrical", apps.electrical),
            ("command_router", apps.command_router),
            ("storage", apps.storage),
            ("downlink", apps.downlink),
            ("mechanical", apps.mechanical),
            ("model_deploy", apps.model_deploy),
        ]
    )
    scheduler.start()
    try:
        threading.Event().wait()  # run until the process is signaled/interrupted
    except KeyboardInterrupt:
        scheduler.stop()
