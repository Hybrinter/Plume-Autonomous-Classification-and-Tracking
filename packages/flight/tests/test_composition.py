"""Tests for the driver-agnostic composition root (build_apps)."""

import numpy as np
from flight.core.composition import MONITORED_SUBSYSTEMS, Drivers, SystemApps, build_apps
from flight.electrical.app import ElectricalApp
from flight.fault.app import FaultApp
from flight.hal.drivers_sim import SimGimbal, SimScalarSensor, SimSensor, SimStationLink
from flight.iss_iface.app import IssIfaceApp
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.time import ManualClock
from flight.payload.app import PayloadApp
from flight.payload.calibration_io import build_identity_calibration
from flight.payload.model import ScriptedDetector
from flight.payload.preprocess import MosaicCalibration
from flight.thermal.app import ThermalApp


def _drivers() -> Drivers:
    """Bundle sim drivers + a scripted detector for composition testing."""
    return Drivers(
        sensor=SimSensor([]),
        gimbal=SimGimbal(),
        detector=ScriptedDetector(np.zeros((256, 256), dtype=np.float32)),
        station=SimStationLink([]),
        thermal_sensor=SimScalarSensor([20.0]),
        power_sensor=SimScalarSensor([10.0]),
    )


def _calib() -> MosaicCalibration:
    """Identity mosaic calibration sized to the default 512x512 sensor geometry."""
    return build_identity_calibration(512, 512)


def test_build_apps_wires_all_five_subsystems() -> None:
    """build_apps constructs all five subsystem apps over the shared bus/clock."""
    apps = build_apps(
        PactConfig(), MessageBus(), ManualClock(), _drivers(), MONITORED_SUBSYSTEMS, _calib()
    )
    assert isinstance(apps, SystemApps)
    assert isinstance(apps.payload, PayloadApp)
    assert isinstance(apps.fault, FaultApp)
    assert isinstance(apps.iss_iface, IssIfaceApp)
    assert isinstance(apps.thermal, ThermalApp)
    assert isinstance(apps.electrical, ElectricalApp)


def test_monitored_subsystems_are_the_heartbeat_producers() -> None:
    """The default monitored set is exactly the four heartbeat-emitting subsystems."""
    assert set(MONITORED_SUBSYSTEMS) == {"payload", "iss_iface", "thermal", "electrical"}


def test_build_apps_shares_one_bus() -> None:
    """All apps are wired to the same bus instance passed in."""
    bus = MessageBus()
    apps = build_apps(PactConfig(), bus, ManualClock(), _drivers(), MONITORED_SUBSYSTEMS, _calib())
    assert apps.payload.bus is bus
    assert apps.fault.bus is bus
    assert apps.thermal.bus is bus
