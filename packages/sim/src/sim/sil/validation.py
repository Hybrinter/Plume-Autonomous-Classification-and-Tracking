"""General env-driven validation-harness API: build + step a flight system for any profile.

This is the composition-root surface the GSE in-process backend drives. GSE imports only
flight.libs and sim, so sim.sil exposes a general validation harness here rather than letting
GSE touch flight.core/flight.payload/flight.fault directly. build_validation_system honors the
full PactConfig.drivers axis vector via flight.core.select_drivers -- so for a sil-link-real
profile (link="real") it yields a RealStationLink while every other axis stays a sim stand-in.

Unlike build_sil_system (which forces an all-"sim" driver config for the deterministic SIL), this
builder is driver-driven: it passes config.drivers through untouched. The returned
ValidationSystem is Protocol-typed (HAL Protocols only), so it carries whatever concrete drivers
the axes selected without the holder ever naming a concrete driver.

ValidationHarness is the general single-threaded stepper: it reuses sim.sil.stepping.step_once
(the one source of truth for a cycle) and threads the payload PayloadState + FDIR watchdog entries
in and out, exactly as SilHarness does, but over the Protocol-typed ValidationSystem.

Contains:
  - ValidationSystem: Protocol-typed holder of the wired apps + bus/clock + selected drivers.
  - build_validation_system: env-driven builder (select_drivers -> build_apps) for any profile.
  - ValidationHarness: deterministic single-threaded stepper (step / run_steps).
  - load_profile_config: load default.toml + a profile override into a PactConfig (raises on Err).

Satisfies: REQ-OPER-HIGH-002.
"""

from __future__ import annotations

# stdlib
import tempfile
from dataclasses import dataclass, replace
from typing import Protocol

# internal
from flight.core.composition import MONITORED_SUBSYSTEMS, SystemApps, build_apps
from flight.core.config_loader import load_config
from flight.core.select_drivers import SimDriverInputs, select_drivers
from flight.fault.watchdog import WatchdogEntry
from flight.hal.interfaces import GimbalActuator, ImagingSensor, ScalarSensor, StationLink
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.messages import SystemModeActivatedMsg
from flight.libs.time import ManualClock
from flight.libs.types import MessageType, Ok, SystemMode
from flight.payload.calibration_io import build_identity_calibration
from flight.payload.graphs.base import GraphId
from flight.payload.lifecycle import InitializationVerifier
from flight.payload.state import PayloadState, graph_name_of, node_name_of

from sim.sil.environment_bind import SilEnvironmentBind
from sim.sil.stepping import step_once


@dataclass(frozen=True)
class ValidationSystem:
    """The wired flight system: apps + shared bus/clock + the env-selected HAL drivers.

    Protocol-typed throughout (flight.hal.interfaces), so it carries whatever concrete
    drivers the PactConfig.drivers axes selected -- a SimSensor or a RealSensor, a
    SimStationLink or a RealStationLink -- without the holder naming a concrete type. This
    is what lets the GSE in-process backend drive any profile through sim only. frozen=True
    without slots is intentional: a holder of Protocol-typed fields does not need slots.
    """

    apps: SystemApps
    bus: MessageBus
    clock: ManualClock
    sensor: ImagingSensor
    gimbal: GimbalActuator
    station: StationLink
    thermal_sensor: ScalarSensor
    power_sensor: ScalarSensor


def build_validation_system(
    config: PactConfig,
    clock: ManualClock,
    sim_inputs: SimDriverInputs | None = None,
    uplink_key: bytes = b"sil-test-key-0000000000000000000",
    activation_epoch: str = "sil",
    *,
    initialization_verifier: InitializationVerifier | None = None,
) -> ValidationSystem:
    """Wire the flight apps over the env-selected drivers on a fresh bus, for any profile.

    Constructs a fresh MessageBus, resolves config.drivers to a concrete Drivers bundle
    via flight.core.select_drivers (the one driver-driven selection path the flight entry uses),
    builds an identity MosaicCalibration sized to the sensor config, and wires every app via
    the driver-agnostic build_apps. Unlike build_sil_system, the driver axis vector is
    passed through untouched, so a 'real' axis yields the real driver (e.g. link="real" ->
    RealStationLink).

    Args:
        config: The validated PactConfig; its driver axes drive driver selection.
        clock: The ManualClock shared by all apps (timestamps; the harness advances `now`).
        sim_inputs: The sim construction inputs (frames, detector, packets, readings);
            required by select_drivers when any selected axis is 'sim'.
        uplink_key: The HMAC-SHA256 secret the iss_iface app uses to authenticate inbound
            TC packets. Defaults to a fixed test key; pass explicitly in command-path tests.
        initialization_verifier: Optional deterministic INIT verifier seam forwarded to
            the payload app; None keeps the pending-by-default production verifier.

    Returns:
        A ValidationSystem holding the wired apps, the shared bus/clock, and the
        Protocol-typed drivers select_drivers resolved from the driver axes.

    Notes:
        The returned drivers fields are the exact Drivers.* objects select_drivers built, so
        the holder needs no cast: the Drivers fields are already declared with the same HAL
        Protocols ValidationSystem uses.

        Storage is redirected to a fresh temp directory so the deterministic in-process harness
        is hermetic (no repo pollution) and isolated per build; the flight entry keeps the
        configured data_root.

        The builder passes ``synchronous_lifecycle=True``. INIT effects run on the
        control thread inside each poll, and the lifecycle daemon does not start.
        SilHarness and the GSE in-process backend share this ManualClock path.
    """
    config = replace(
        config,
        storage=replace(config.storage, data_root=tempfile.mkdtemp(prefix="pact-sil-storage-")),
    )
    bus = MessageBus()
    drivers = select_drivers(config, clock, sim_inputs)
    calib = build_identity_calibration(config.sensor.height_px, config.sensor.width_px)
    apps = build_apps(
        config,
        bus,
        clock,
        drivers,
        MONITORED_SUBSYSTEMS,
        calib,
        uplink_key,
        activation_epoch,
        initialization_verifier=initialization_verifier,
        synchronous_lifecycle=True,
    )
    return ValidationSystem(
        apps=apps,
        bus=bus,
        clock=clock,
        sensor=drivers.sensor,
        gimbal=drivers.gimbal,
        station=drivers.station,
        thermal_sensor=drivers.thermal_sensor,
        power_sensor=drivers.power_sensor,
    )


class ValidationHarness:
    """Deterministic single-threaded driver for a ValidationSystem (no scheduler threads)."""

    def __init__(self, system: ValidationSystem, bind: SilEnvironmentBind | None = None) -> None:
        """Seed the payload control state and the FDIR watchdog entries.

        Args:
            system: The wired ValidationSystem to drive.
            bind: Optional world evaluate + driver feed run inside each step_once
                after loop catch-up and before acquire.
        """
        self._system = system
        self._bind = bind
        self._now = system.clock.monotonic_s()
        self._payload_state: PayloadState = system.apps.payload.initial_state()
        self._fault_entries: dict[str, WatchdogEntry] = system.apps.fault.initial_entries()

    def payload_graph(self) -> str | None:
        """Return the active payload GraphId value, or None before activation."""
        name = graph_name_of(self._payload_state)
        return name if name else None

    def payload_node(self) -> str | None:
        """Return the active graph node value, or None before activation."""
        name = node_name_of(self._payload_state)
        return name if name else None

    def payload_system_mode(self) -> SystemMode | None:
        """Return the system mode of the last accepted activation, or None.

        Derived from the accepted SystemModeActivatedMsg snapshot only; a fault
        latch or a mode request never appears here.
        """
        last = self._payload_state.activation.last
        if last is None:
            return None
        return _GRAPH_TO_MODE[last.graph_id]

    def step(self, now: float) -> None:
        """Advance every subsystem one cycle over the shared bus (delegates to step_once).

        Args:
            now: Target monotonic seconds for the graphs and watchdog;
                step_once advances the shared clock forward to it.
        """
        self._now = now
        system = self._system
        self._payload_state, self._fault_entries = step_once(
            system.apps,
            system.sensor,
            system.gimbal,
            system.bus,
            system.clock,
            now,
            self._payload_state,
            self._fault_entries,
            bind=self._bind,
        )

    def run_steps(self, count: int, dt: float = 1.0) -> None:
        """Run count deterministic steps, advancing `now` by dt each step.

        step_once owns the shared ManualClock: each step advances it forward
        to the step's `now`, so callers never advance the clock separately.
        For real drivers the advanced `now` still drives the graphs and
        watchdog deterministically.

        Args:
            count: Number of steps to run.
            dt: Seconds to advance `now` per step.
        """
        now = self._now
        for _ in range(count):
            now += dt
            self.step(now)


def load_profile_config(config_path: str, override_path: str) -> PactConfig:
    """Load config_path merged with a profile override into a PactConfig, raising on failure.

    A composition-root convenience: a config load failure at startup is unrecoverable, so this
    raises rather than returning a Result (per the Result-vs-startup-exception distinction).

    Args:
        config_path: Path to the base TOML config (typically "config/default.toml").
        override_path: Path to the deployment-profile override TOML (e.g. "profiles/sil.toml").

    Returns:
        The merged, validated PactConfig.

    Raises:
        ValueError: If load_config returns an Err (missing file, parse, or validation error).
    """
    result = load_config(config_path, override_path)
    if not isinstance(result, Ok):
        raise ValueError(f"config load failed: {result.error}")
    return result.value


_GRAPH_TO_MODE: dict[GraphId, SystemMode] = {
    GraphId.IDLE: SystemMode.IDLE,
    GraphId.STOW: SystemMode.STOW,
    GraphId.SAFE: SystemMode.SAFE,
    GraphId.INIT: SystemMode.INIT,
    GraphId.OPERATE: SystemMode.OPERATE,
}


class ActivationTarget(Protocol):
    """Structural holder for explicit activation injection (SilSystem/ValidationSystem)."""

    @property
    def apps(self) -> SystemApps: ...
    @property
    def bus(self) -> MessageBus: ...
    @property
    def clock(self) -> ManualClock: ...


def publish_activation(
    system: ActivationTarget,
    mode: SystemMode,
    sequence: int = 1,
    previous_mode: SystemMode | None = None,
    request_id: str | None = None,
    recovery_authorized: bool = False,
) -> None:
    """Publish an explicit authority activation onto the system bus.

    Explicit test/scenario fixture only: this bypasses the real authority's
    arbitration and is not acceptance proof. The injected record is still
    published verbatim, but the live system-mode authority is seeded coherently
    first -- when the record's epoch matches the authority epoch and its
    sequence is strictly newer than the authority's counter, the authority's
    active snapshot and sequence are advanced to it, so the authority neither
    boots a contradictory SAFE nor reuses the sequence. A stale or
    foreign-epoch injection never rewrites the seed (that keeps the
    stale/conflict injection tests meaningful) and never lowers the counter.

    Args:
        system: The wired ValidationSystem whose bus carries the record.
        mode: The activated SystemMode (maps onto the same-named payload graph).
        sequence: Authority activation sequence under the system epoch.
        previous_mode: Previously active mode, or None on initial activation.
        request_id: Correlated request identity, or None.
        recovery_authorized: Authority-approved EXIT_SAFE recovery flag.
    """
    msg = SystemModeActivatedMsg(
        msg_type=MessageType.SYSTEM_MODE_ACTIVATED,
        timestamp_utc=system.clock.wall_clock_iso(),
        epoch=system.apps.payload.activation_epoch,
        sequence=sequence,
        previous_mode=previous_mode,
        active_mode=mode,
        reason="injected_activation",
        request_id=request_id,
        recovery_authorized=recovery_authorized,
    )
    authority = system.apps.system_modes
    state = authority.state
    if msg.epoch == authority.epoch and msg.sequence > state.sequence:
        state.active = msg
        state.sequence = msg.sequence
    system.bus.publish(msg)
