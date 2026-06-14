"""Thermal housekeeping app: temperature telemetry, over-limit fault, command ack.

Minimal subsystem app proving the thermal node is in the topology: each cycle it
samples a temperature via a ScalarSensor, publishes a TelemetryEventMsg, and publishes
a THERMAL_OVER_LIMIT FaultEventMsg when the reading exceeds cfg.thermal_limit_c. It
acknowledges any CommandMsg targeting "thermal" with a command_ack telemetry event, and
emits periodic heartbeats. All decision logic is trivial; time is injected via Clock.

Satisfies: REQ-SAFE-HIGH-002 (thermal self-reporting), REQ-OPER-HIGH-002 (subsystem app).
"""

from __future__ import annotations

# stdlib
import threading
from dataclasses import dataclass, field

# internal
from flight.hal.interfaces import ScalarSensor
from flight.libs.bus import MessageBus, Subscription
from flight.libs.config import FaultConfig, PactConfig
from flight.libs.messages import (
    CommandAckMsg,
    FaultEventMsg,
    HeartbeatMsg,
    RoutedCommandMsg,
    TelemetryEventMsg,
)
from flight.libs.time import Clock
from flight.libs.types import AckStatus, FaultCode, MessageType, Ok

SUBSYSTEM = "thermal"
_SET_THERMAL_LIMIT = "SET_THERMAL_LIMIT"


@dataclass(slots=True)
class ThermalState:
    """Mutable thermal-app state set by executed commands.

    Fields:
        limit_c_override: Ground-commanded over-limit threshold (Celsius); when set it
            supersedes cfg.thermal_limit_c in sample(). None means use the config default.
    """

    limit_c_override: float | None = None


@dataclass(frozen=True)
class ThermalApp:
    """Thermal housekeeping subsystem app (telemetry + over-limit fault + commandable)."""

    cfg: FaultConfig
    bus: MessageBus
    clock: Clock
    sensor: ScalarSensor
    commands: Subscription[RoutedCommandMsg]
    state: ThermalState = field(default_factory=ThermalState)

    @staticmethod
    def from_config(
        cfg: PactConfig,
        bus: MessageBus,
        clock: Clock,
        sensor: ScalarSensor,
    ) -> ThermalApp:
        """Assemble a ThermalApp and subscribe it to routed commands.

        Args:
            cfg: Top-level PactConfig (cfg.fault is retained for the limit + heartbeat).
            bus: The MessageBus to publish onto and subscribe to.
            clock: Injected Clock.
            sensor: The ScalarSensor reading temperature in Celsius.

        Returns:
            A ThermalApp holding a fresh RoutedCommandMsg subscription and cleared state.
        """
        return ThermalApp(
            cfg=cfg.fault,
            bus=bus,
            clock=clock,
            sensor=sensor,
            commands=bus.subscribe(RoutedCommandMsg),
            state=ThermalState(),
        )

    def sample(self) -> None:
        """Read the temperature, publish telemetry, and emit a fault if over the limit.

        On a sensor read error the cycle is skipped (no telemetry, no fault) -- a
        transient read failure surfaces as missing telemetry, which the watchdog/ground
        observe; there is no dedicated sensor-fault code.
        """
        result = self.sensor.read()
        if not isinstance(result, Ok):
            return
        temperature_c = result.value
        limit_c = self.state.limit_c_override
        if limit_c is None:
            limit_c = self.cfg.thermal_limit_c
        self.bus.publish(
            TelemetryEventMsg(
                msg_type=MessageType.TELEMETRY_EVENT,
                timestamp_utc=self.clock.wall_clock_iso(),
                subsystem=SUBSYSTEM,
                event_name="thermal_sample",
                payload={"temperature_c": temperature_c},
            )
        )
        if temperature_c > limit_c:
            self.bus.publish(
                FaultEventMsg(
                    msg_type=MessageType.FAULT_EVENT,
                    timestamp_utc=self.clock.wall_clock_iso(),
                    fault_code=FaultCode.THERMAL_OVER_LIMIT,
                    subsystem=SUBSYSTEM,
                    detail=(f"temperature {temperature_c:.1f}C exceeds limit {limit_c:.1f}C"),
                )
            )

    def handle_commands(self) -> None:
        """Execute each routed command targeting this subsystem and emit an execution ack.

        SET_THERMAL_LIMIT applies a new over-limit threshold (stored in ThermalState and used
        by the next sample) and acks ACCEPTED; any other opcode targeting thermal acks
        REJECTED (the router routed it here, but thermal does not implement it).
        """
        while not self.commands.empty():
            command = self.commands.get_nowait()
            if command.target != SUBSYSTEM:
                continue
            if command.command_id == _SET_THERMAL_LIMIT:
                self.state.limit_c_override = float(command.params["limit_c"])
                self._publish_ack(command, AckStatus.ACCEPTED, FaultCode.NONE, "thermal limit set")
            else:
                self._publish_ack(
                    command, AckStatus.REJECTED, FaultCode.COMMAND_INVALID, "unsupported command"
                )

    def _publish_ack(
        self, command: RoutedCommandMsg, status: AckStatus, fault: FaultCode, detail: str
    ) -> None:
        """Publish an execution CommandAckMsg correlated to a routed command."""
        self.bus.publish(
            CommandAckMsg(
                msg_type=MessageType.COMMAND_ACK,
                timestamp_utc=self.clock.wall_clock_iso(),
                status=status,
                command_id=command.command_id,
                source=command.source,
                seq=command.seq,
                fault_code=fault,
                detail=detail,
            )
        )

    def run(self, stop_event: threading.Event) -> None:
        """Run the housekeeping loop until stop_event is set, with periodic heartbeats.

        Each iteration handles commands, samples, and emits a heartbeat every
        cfg.watchdog_interval_s; then waits one interval.

        Args:
            stop_event: threading.Event; the loop exits cleanly once it is set.
        """
        sequence = 0
        last_heartbeat = self.clock.monotonic_s()
        while not stop_event.is_set():
            self.handle_commands()
            self.sample()
            now = self.clock.monotonic_s()
            if now - last_heartbeat >= self.cfg.watchdog_interval_s:
                self.bus.publish(
                    HeartbeatMsg(
                        msg_type=MessageType.HEARTBEAT,
                        timestamp_utc=self.clock.wall_clock_iso(),
                        subsystem=SUBSYSTEM,
                        sequence=sequence,
                    )
                )
                sequence += 1
                last_heartbeat = now
            stop_event.wait(timeout=self.cfg.watchdog_interval_s)
