"""Driver-agnostic composition root: wires the subsystem apps over one bus + clock.

build_apps() is the single place the full app topology is assembled. It depends only
on HAL Protocols and the apps -- never on concrete drivers -- so the same wiring serves
the flight entry (real drivers, in core/main.py) and the SIL (sim drivers, in
sim/sil). The caller constructs the Drivers bundle and owns the bus and clock.

Contains:
  - Drivers: the bundle of injected HAL drivers + the detector backend.
  - SystemApps: the five constructed subsystem apps.
  - MONITORED_SUBSYSTEMS: the heartbeat-emitting subsystems the FDIR watchdog watches.
  - build_apps: construct every app from config + bus + clock + drivers.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass

# internal
from flight.electrical.app import ElectricalApp
from flight.fault.app import FaultApp
from flight.hal.interfaces import GimbalActuator, ImagingSensor, ScalarSensor, StationLink
from flight.iss_iface.app import IssIfaceApp
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.time import Clock
from flight.payload.app import PayloadApp
from flight.payload.model import DetectorBackend
from flight.payload.preprocess import MosaicCalibration
from flight.thermal.app import ThermalApp

# The subsystems that run persistent loops and emit heartbeats; the FDIR watchdog
# monitors exactly these (the fault subsystem does not monitor itself).
MONITORED_SUBSYSTEMS: tuple[str, ...] = ("payload", "iss_iface", "thermal", "electrical")


@dataclass(frozen=True)
class Drivers:
    """Bundle of injected HAL drivers + the detector backend for one composition.

    The composition root (flight entry or SIL) constructs the concrete implementations;
    build_apps consumes only the Protocol types.
    """

    sensor: ImagingSensor
    gimbal: GimbalActuator
    detector: DetectorBackend
    station: StationLink
    thermal_sensor: ScalarSensor
    power_sensor: ScalarSensor


@dataclass(frozen=True)
class SystemApps:
    """The five constructed subsystem apps, sharing one bus and clock."""

    payload: PayloadApp
    fault: FaultApp
    iss_iface: IssIfaceApp
    thermal: ThermalApp
    electrical: ElectricalApp


def build_apps(
    config: PactConfig,
    bus: MessageBus,
    clock: Clock,
    drivers: Drivers,
    monitored: tuple[str, ...],
    calib: MosaicCalibration,
) -> SystemApps:
    """Construct every subsystem app wired to the shared bus and clock.

    Args:
        config: The validated PactConfig.
        bus: The single MessageBus all apps publish to / subscribe from.
        clock: The injected Clock (RealClock in flight, ManualClock in SIL/tests).
        drivers: The HAL driver bundle (real or sim) plus the detector backend.
        monitored: Subsystem names the FDIR watchdog should watch (use MONITORED_SUBSYSTEMS).
        calib: The MosaicCalibration the payload app applies to the raw mosaic plane
            (loaded from artifacts in flight; identity in SIL). Constructed by the
            composition root and injected here so build_apps stays driver-agnostic.

    Returns:
        A SystemApps with all five apps constructed.
    """
    return SystemApps(
        payload=PayloadApp.from_config(
            config, drivers.sensor, drivers.gimbal, drivers.detector, bus, clock, calib
        ),
        fault=FaultApp.from_config(config, bus, clock, monitored),
        iss_iface=IssIfaceApp.from_config(config, bus, clock, drivers.station),
        thermal=ThermalApp.from_config(config, bus, clock, drivers.thermal_sensor),
        electrical=ElectricalApp.from_config(config, bus, clock, drivers.power_sensor),
    )
