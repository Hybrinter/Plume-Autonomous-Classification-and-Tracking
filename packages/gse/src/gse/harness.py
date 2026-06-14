"""GSE harness backends: deterministic in-process vs. deferred socket transport.

A HarnessBackend abstracts how a scenario is built, stepped, command-injected, and
captured -- so the orchestrator scores the same assertions regardless of transport.

InProcessBackend is the blessed x86 backend. It builds a PactConfig from the profile
(sim.sil.load_profile_config), renders the plume scene (sim.scene), and drives the wired
flight apps through sim.sil.ValidationHarness over the env-selected drivers (sim.sil owns
the build + step seam, so gse never touches the flight composition root or apps). When
the profile's link axis is "real" it stands up a RealStationLink (chosen by the harness over
a free TCP/UDP port pair) and a GSE StationEmulator as the live counterpart, so the
authenticated command + CCSDS downlink path runs over real sockets in one process.

For an all-sim link, SimStationLink's inbound queue is fixed at construction, so all
scenario commands are pre-baked into SimDriverInputs.inbound_packets at build() time from
the command timeline; inject_command() is then a documented no-op on that path. Because
IssIfaceApp.pump_uplink drains every queued packet on the first tick, sim-link scenarios
do NOT honor CommandStep.at_frame timing (all commands land on step 1); author sim-link
scenarios to be insensitive to command-arrival ordering. Timed delivery is real-link only.

SocketBackend is declared (PIL/HIL transport) but raises NotImplementedError -- those
venues are DEFINED, NOT RUN.

Dependency surface: this module imports flight.libs and sim only (sim.sil exposes the
validation harness so gse never imports the flight composition root or apps directly).
lint-imports enforces the one-way flight/sim !-> gse rule.

Contains:
  - TelemetryCapture: frozen holder of scored bus events + downlink bytes.
  - HarnessBackend: the runtime-checkable backend Protocol.
  - InProcessBackend: deterministic ManualClock + ValidationHarness backend (sim or real link).
  - SocketBackend: deferred PIL/HIL socket backend (NotImplementedError).

Satisfies: REQ-COMM-HIGH-001, REQ-COMM-HIGH-003, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

# stdlib
import socket
from dataclasses import dataclass, replace
from typing import Protocol, TypeVar, runtime_checkable

# internal
from flight.libs.bus import Subscription
from flight.libs.commands import build_tc_packet
from flight.libs.config import LinkConfig
from flight.libs.messages import (
    CommandAckMsg,
    GimbalCommandMsg,
    InferenceResultMsg,
    ModeChangeMsg,
)
from flight.libs.time import ManualClock
from flight.libs.types import AckStatus, Err, SystemMode
from sim.scene import build_frames, plume_detector
from sim.sil import (
    SimDriverInputs,
    ValidationHarness,
    ValidationSystem,
    build_validation_system,
    load_profile_config,
)

from gse.scenario import CommandStep, Scenario
from gse.station import StationEmulator

_SIL_KEY = b"sil-test-key-0000000000000000000"

# Off-origin tolerance (deg) for the gimbal-moved flag. SimGimbal.read_position() adds
# encoder noise at config.gimbal.sim_encoder_noise_deg (default 0.005 deg 1-sigma) on every
# read, so a strict != (0.0, 0.0) test would spuriously report motion even when stationary.
# 0.1 deg is 20x the noise 1-sigma, well below real tracked motion (degrees).
_GIMBAL_MOVED_TOLERANCE_DEG = 0.1

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class TelemetryCapture:
    """Scored telemetry collected from one stepped scenario run.

    Fields:
        inference_count: Number of InferenceResultMsg published over the run.
        gimbal_moved: True if the payload moved the gimbal off the (0, 0) origin by more
            than the encoder-noise tolerance (_GIMBAL_MOVED_TOLERANCE_DEG).
        mode_changes: The SystemMode of every ModeChangeMsg, in publication order.
        acks: The AckStatus of every CommandAckMsg observed on the bus, in order.
        downlink_packets: Raw CCSDS TM datagrams the StationEmulator received over UDP
            (empty for an all-sim link, where no real socket carries downlink).
    """

    inference_count: int
    gimbal_moved: bool
    mode_changes: tuple[SystemMode, ...]
    acks: tuple[AckStatus, ...]
    downlink_packets: tuple[bytes, ...]


@runtime_checkable
class HarnessBackend(Protocol):
    """Transport-agnostic scenario backend the orchestrator drives.

    Implementations build a scenario over a profile, step it, inject commands, collect a
    TelemetryCapture for scoring, and shut down. The orchestrator scores the same
    frame-portable assertions against the capture regardless of which backend produced it.
    """

    def build(self, scenario: Scenario, profile_path: str) -> None:
        """Construct the system for scenario under the config override at profile_path."""
        ...

    def step(self, now: float) -> None:
        """Advance the system one deterministic cycle at monotonic-seconds now."""
        ...

    def inject_command(self, step: CommandStep) -> None:
        """Deliver one scenario command to the system (live over the link, or a no-op)."""
        ...

    def collect(self) -> TelemetryCapture:
        """Drain accumulated telemetry into a frozen TelemetryCapture for scoring."""
        ...

    def shutdown(self) -> None:
        """Release any sockets/threads the backend stood up (idempotent)."""
        ...


def _free_port_pair() -> tuple[int, int]:
    """Reserve two distinct free localhost ports by transient binds.

    Returns:
        tuple[int, int]: A (tcp_port, udp_port) pair that were free at probe time.

    Notes:
        Binds two ephemeral sockets, reads the OS-assigned ports, then closes them. A
        narrow TOCTOU window remains, but localhost in a single CI job makes a collision
        practically impossible; the alternative (passing live sockets) breaks the
        RealStationLink/StationEmulator contract, which both take host+port.
    """
    probes: list[socket.socket] = []
    ports: list[int] = []
    for _ in range(2):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        ports.append(s.getsockname()[1])
        probes.append(s)
    for s in probes:
        s.close()
    return ports[0], ports[1]


def _scenario_packets(scenario: Scenario, key: bytes, tc_apid: int) -> list[bytes]:
    """Build signed TC packets for every command step in a scenario (sim-link pre-bake).

    Args:
        scenario: The scenario whose command timeline to serialize.
        key: The shared HMAC-SHA256 uplink secret.
        tc_apid: The telecommand APID from the resolved LinkConfig.

    Returns:
        list[bytes]: One framed CCSDS TC packet per CommandStep, in timeline order. These
        seed SimStationLink.inbound at build time because its queue is fixed at
        construction (the sim path cannot accept a live mid-run injection). Note that
        pump_uplink drains all of them on the first tick, so at_frame timing is NOT honored
        on the sim link.
    """
    return [
        build_tc_packet(step.command_id, step.params, step.source, step.seq, key, tc_apid)
        for step in scenario.commands
    ]


class InProcessBackend:
    """Deterministic single-process backend: ManualClock + ValidationHarness over the drivers."""

    def __init__(self) -> None:
        """Initialize an unbuilt backend; build() populates the system + harness state."""
        self._clock = ManualClock()
        self._system: ValidationSystem | None = None
        self._harness: ValidationHarness | None = None
        self._emulator: StationEmulator | None = None
        self._inf_sub: Subscription[InferenceResultMsg] | None = None
        self._gimbal_sub: Subscription[GimbalCommandMsg] | None = None
        self._mode_sub: Subscription[ModeChangeMsg] | None = None
        self._ack_sub: Subscription[CommandAckMsg] | None = None
        self._link_real = False

    def build(self, scenario: Scenario, profile_path: str) -> None:
        """Build the wired apps + drivers for scenario under the profile override.

        Args:
            scenario: The scenario (scene spec + command timeline) to realize.
            profile_path: Path to the profile TOML applied as an override over
                config/default.toml (selects the per-axis sim/real environment).

        Notes:
            For a real link axis, frees a TCP/UDP port pair, replaces LinkConfig so
            RealStationLink and StationEmulator agree on the endpoints, builds the system
            (the harness selects RealStationLink over those ports), and connects a
            StationEmulator as the live counterpart so AOS holds and downlink datagrams are
            captured. For a sim link, the command timeline is pre-baked into the
            SimStationLink inbound queue (its queue is fixed at construction). Because
            pump_uplink drains every queued packet on the first tick, sim-link
            CommandStep.at_frame timing is NOT honored (all commands ingest on step 1);
            sim-link scenarios must be order-insensitive. Subscriptions are created BEFORE
            any step so no published message is missed (the bus only delivers to live subs).
        """
        config = load_profile_config("config/default.toml", profile_path)
        self._link_real = config.environment.link == "real"

        frames = build_frames(scenario.scene.num_frames, scenario.scene.seed)
        detector = plume_detector()

        if self._link_real:
            tcp_port, udp_port = _free_port_pair()
            link_cfg: LinkConfig = replace(
                config.link, command_tcp_port=tcp_port, telemetry_udp_port=udp_port
            )
            config = replace(config, link=link_cfg)
            sim_inputs = SimDriverInputs(
                frames=frames,
                detector=detector,
                inbound_packets=[],
                thermal_readings=list(scenario.scene.thermal_readings),
                power_readings=list(scenario.scene.power_readings),
            )
            system = build_validation_system(config, self._clock, sim_inputs, _SIL_KEY)
            self._emulator = StationEmulator(
                tcp_host=link_cfg.command_tcp_host,
                tcp_port=link_cfg.command_tcp_port,
                udp_host=link_cfg.telemetry_udp_host,
                udp_port=link_cfg.telemetry_udp_port,
                key=_SIL_KEY,
                tc_apid=link_cfg.tc_apid,
            )
            self._emulator.connect()
        else:
            inbound = _scenario_packets(scenario, _SIL_KEY, config.link.tc_apid)
            sim_inputs = SimDriverInputs(
                frames=frames,
                detector=detector,
                inbound_packets=inbound,
                thermal_readings=list(scenario.scene.thermal_readings),
                power_readings=list(scenario.scene.power_readings),
            )
            system = build_validation_system(config, self._clock, sim_inputs, _SIL_KEY)

        self._inf_sub = system.bus.subscribe(InferenceResultMsg)
        self._gimbal_sub = system.bus.subscribe(GimbalCommandMsg)
        self._mode_sub = system.bus.subscribe(ModeChangeMsg)
        self._ack_sub = system.bus.subscribe(CommandAckMsg)

        self._system = system
        self._harness = ValidationHarness(system)

    def step(self, now: float) -> None:
        """Advance every subsystem one cycle via the harness, advancing the ManualClock first.

        Args:
            now: Monotonic seconds for the arbiter/watchdog (caller-advanced per step).

        Notes:
            The shared ManualClock is advanced to now so SimGimbal first-order dynamics
            integrate between steps (the closed loop only moves the gimbal across steps).
        """
        if self._harness is None:
            raise RuntimeError("build() must be called before step()")
        delta = now - self._clock.monotonic_s()
        if delta > 0.0:
            self._clock.advance(delta)
        self._harness.step(now)

    def inject_command(self, step: CommandStep) -> None:
        """Send one command live for a real link; a no-op (pre-baked) for a sim link.

        Args:
            step: The command timeline entry to deliver.

        Notes:
            Sim-link commands are pre-baked into SimStationLink.inbound at build() time
            (its queue is fixed at construction), so this is intentionally a no-op there.
            Real-link commands are uplinked over TCP through the StationEmulator.
        """
        if self._link_real:
            if self._emulator is None:
                raise RuntimeError("real link backend has no StationEmulator")
            self._emulator.send_command(step.command_id, step.params, step.source, step.seq)

    def collect(self) -> TelemetryCapture:
        """Drain subscriptions + emulator UDP into a TelemetryCapture for scoring.

        Returns:
            TelemetryCapture: inference count, gimbal-moved flag, ordered mode changes,
            ordered ack statuses, and (real link only) the downlink datagrams the
            StationEmulator received.

        Notes:
            gimbal_moved compares the driver's authoritative read_position() against the
            origin with a tolerance (_GIMBAL_MOVED_TOLERANCE_DEG) that swamps SimGimbal's
            per-read encoder noise, so a stationary gimbal reliably reports False.
        """
        if (
            self._system is None
            or self._inf_sub is None
            or self._gimbal_sub is None
            or self._mode_sub is None
            or self._ack_sub is None
        ):
            raise RuntimeError("build() must be called before collect()")

        inference_count = len(self._drain(self._inf_sub))

        self._drain(self._gimbal_sub)
        gimbal_moved = False
        read = self._system.gimbal.read_position()
        if not isinstance(read, Err):
            worst = max(abs(read.value.az_deg), abs(read.value.el_deg))
            gimbal_moved = worst > _GIMBAL_MOVED_TOLERANCE_DEG

        mode_changes = tuple(m.new_mode for m in self._drain(self._mode_sub))
        acks = tuple(a.status for a in self._drain(self._ack_sub))

        downlink: tuple[bytes, ...] = ()
        if self._emulator is not None:
            downlink = tuple(self._emulator.poll_downlink(timeout_s=0.2))

        return TelemetryCapture(
            inference_count=inference_count,
            gimbal_moved=gimbal_moved,
            mode_changes=mode_changes,
            acks=acks,
            downlink_packets=downlink,
        )

    def shutdown(self) -> None:
        """Close the StationEmulator and the wired flight apps' link (idempotent)."""
        if self._emulator is not None:
            self._emulator.close()
            self._emulator = None
        if self._system is not None:
            self._system.station.close()

    @staticmethod
    def _drain(subscription: Subscription[_T]) -> list[_T]:
        """Drain all pending messages from a subscription into a list (order-preserving)."""
        out: list[_T] = []
        while not subscription.empty():
            out.append(subscription.get_nowait())
        return out


class SocketBackend:
    """Deferred PIL/HIL transport backend: declared by the matrix, not implemented."""

    def build(self, scenario: Scenario, profile_path: str) -> None:
        """Raise: the PIL/HIL socket backend is defined-not-run for this milestone."""
        raise NotImplementedError("PIL/HIL socket backend deferred")

    def step(self, now: float) -> None:
        """Raise: the PIL/HIL socket backend is defined-not-run for this milestone."""
        raise NotImplementedError("PIL/HIL socket backend deferred")

    def inject_command(self, step: CommandStep) -> None:
        """Raise: the PIL/HIL socket backend is defined-not-run for this milestone."""
        raise NotImplementedError("PIL/HIL socket backend deferred")

    def collect(self) -> TelemetryCapture:
        """Raise: the PIL/HIL socket backend is defined-not-run for this milestone."""
        raise NotImplementedError("PIL/HIL socket backend deferred")

    def shutdown(self) -> None:
        """Raise: the PIL/HIL socket backend is defined-not-run for this milestone."""
        raise NotImplementedError("PIL/HIL socket backend deferred")
