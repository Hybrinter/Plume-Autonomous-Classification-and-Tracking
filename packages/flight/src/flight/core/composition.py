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
from flight.core.command_router import CommandRouter
from flight.core.downlink import DownlinkManager
from flight.core.model_deploy import ModelDeployService
from flight.core.storage import StorageService
from flight.electrical.app import ElectricalApp
from flight.fault.app import FaultApp
from flight.hal.interfaces import (
    GimbalActuator,
    ImagingSensor,
    LaunchLock,
    ScalarSensor,
    StationLink,
)
from flight.iss_iface.app import IssIfaceApp
from flight.libs.bus import MessageBus, OverflowPolicy, QueuePolicy
from flight.libs.config import PactConfig
from flight.libs.messages import (
    CommandAckMsg,
    CommandMsg,
    DownlinkItemMsg,
    FaultEventMsg,
    GimbalCommandMsg,
    HeartbeatMsg,
    InferenceResultMsg,
    LaunchLockStateMsg,
    LinkStateMsg,
    ModeChangeMsg,
    ModelDeployStateMsg,
    ModelStagedMsg,
    ProcessedFrameMsg,
    ProductRefMsg,
    RoutedCommandMsg,
    SafetyStateMsg,
    StorageWriteMsg,
    TelemetryEventMsg,
    UploadChunkMsg,
)
from flight.libs.time import Clock
from flight.mechanical.app import MechanicalApp
from flight.payload.app import PayloadApp
from flight.payload.model import DetectorBackend
from flight.payload.preprocess import MosaicCalibration
from flight.thermal.app import ThermalApp

# The subsystems that run persistent loops and emit heartbeats; the FDIR watchdog
# monitors exactly these (the fault subsystem does not monitor itself).
MONITORED_SUBSYSTEMS: tuple[str, ...] = (
    "payload",
    "iss_iface",
    "thermal",
    "electrical",
    "command_router",
    "storage",
    "downlink",
    "mechanical",
    "model_deploy",
)


# Per-message-type bus queue bounds (spec Section 7). Commands/faults/acks/mode/uploads are
# NEVER_DROP (losing one is never acceptable -- a soft-bound exceedance is counted as an
# anomaly); high-rate telemetry/products are DROP_OLDEST (shed stale data under backpressure,
# keep the loop flowing). Bounds are generous; the deterministic SIL keeps an unbounded bus.
_NEVER_DROP_BOUND = 1024
_DROP_OLDEST_BOUND = 8192


def default_bus_policy() -> dict[type, QueuePolicy]:
    """Build the flight bus queue policy: NEVER_DROP for commands/faults, DROP_OLDEST for telemetry.

    Returns:
        A dict mapping each message type to its QueuePolicy. Used by flight.core.main; the SIL
        composition root leaves the bus unbounded (default) for determinism.
    """
    never = QueuePolicy(maxsize=_NEVER_DROP_BOUND, overflow=OverflowPolicy.NEVER_DROP)
    drop = QueuePolicy(maxsize=_DROP_OLDEST_BOUND, overflow=OverflowPolicy.DROP_OLDEST)
    policy: dict[type, QueuePolicy] = {}
    for never_type in (
        CommandMsg,
        RoutedCommandMsg,
        CommandAckMsg,
        FaultEventMsg,
        ModeChangeMsg,
        ModelStagedMsg,
        UploadChunkMsg,
        StorageWriteMsg,
    ):
        policy[never_type] = never
    for drop_type in (
        TelemetryEventMsg,
        ProcessedFrameMsg,
        InferenceResultMsg,
        LinkStateMsg,
        LaunchLockStateMsg,
        GimbalCommandMsg,
        HeartbeatMsg,
        ProductRefMsg,
        DownlinkItemMsg,
        ModelDeployStateMsg,
        SafetyStateMsg,
    ):
        policy[drop_type] = drop
    return policy


@dataclass(frozen=True)
class Drivers:
    """Bundle of injected HAL drivers + the detector backend for one composition.

    The composition root (flight entry or SIL) constructs the concrete implementations;
    build_apps consumes only the Protocol types. The launch_lock is always a SimLaunchLock
    today (no real driver exists -- the device is hardware-deferred, a permanent VCRM gap).
    """

    sensor: ImagingSensor
    gimbal: GimbalActuator
    detector: DetectorBackend
    station: StationLink
    thermal_sensor: ScalarSensor
    power_sensor: ScalarSensor
    launch_lock: LaunchLock


@dataclass(frozen=True)
class SystemApps:
    """The constructed subsystem apps + core services, sharing one bus and clock.

    The five subsystem apps (payload/fault/iss_iface/thermal/electrical) plus the core-hosted
    command_router service (spec Section 10 Approach A). All are constructed by build_apps,
    run by the flight Scheduler, and stepped by the deterministic SIL/GSE harness.
    """

    payload: PayloadApp
    fault: FaultApp
    iss_iface: IssIfaceApp
    thermal: ThermalApp
    electrical: ElectricalApp
    command_router: CommandRouter
    storage: StorageService
    downlink: DownlinkManager
    mechanical: MechanicalApp
    model_deploy: ModelDeployService


def build_apps(
    config: PactConfig,
    bus: MessageBus,
    clock: Clock,
    drivers: Drivers,
    monitored: tuple[str, ...],
    calib: MosaicCalibration,
    uplink_key: bytes,
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
        uplink_key: The shared HMAC-SHA256 secret for authenticating inbound TC packets.
            Loaded from disk by the composition root and injected here so build_apps
            and the apps themselves stay key-file-agnostic.

    Returns:
        A SystemApps with all five apps constructed.
    """
    storage = StorageService.from_config(config, bus, clock)
    return SystemApps(
        payload=PayloadApp.from_config(
            config, drivers.sensor, drivers.gimbal, drivers.detector, bus, clock, calib, storage
        ),
        fault=FaultApp.from_config(config, bus, clock, monitored),
        iss_iface=IssIfaceApp.from_config(
            config, bus, clock, drivers.station, uplink_key, storage, storage
        ),
        thermal=ThermalApp.from_config(config, bus, clock, drivers.thermal_sensor),
        electrical=ElectricalApp.from_config(config, bus, clock, drivers.power_sensor),
        command_router=CommandRouter.from_config(config, bus, clock),
        storage=storage,
        downlink=DownlinkManager.from_config(config, bus, clock),
        mechanical=MechanicalApp.from_config(config, bus, clock, drivers.launch_lock),
        model_deploy=ModelDeployService.from_config(config, bus, clock, storage),
    )
