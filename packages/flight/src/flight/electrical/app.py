"""Electrical housekeeping app: power telemetry, over-limit fault, command ack.

Minimal subsystem app proving the electrical node is in the topology: each cycle it
samples a power draw via a ScalarSensor, publishes a TelemetryEventMsg, and publishes a
POWER_OVER_LIMIT FaultEventMsg when the reading exceeds cfg.power_limit_w. It
acknowledges any CommandMsg targeting "electrical" with a command_ack telemetry event,
and emits periodic heartbeats. All decision logic is trivial; time is injected via Clock.

Satisfies: REQ-SAFE-HIGH-002 (power self-reporting), REQ-OPER-HIGH-002 (subsystem app).
"""

from __future__ import annotations

# stdlib
import threading
from dataclasses import dataclass

# internal
from flight.hal.interfaces import ScalarSensor
from flight.libs.bus import MessageBus, Subscription
from flight.libs.config import FaultConfig, PactConfig
from flight.libs.messages import CommandMsg, FaultEventMsg, HeartbeatMsg, TelemetryEventMsg
from flight.libs.time import Clock
from flight.libs.types import FaultCode, MessageType, Ok

SUBSYSTEM = "electrical"


@dataclass(frozen=True)
class ElectricalApp:
    """Electrical housekeeping subsystem app (telemetry + over-limit fault + commandable)."""

    cfg: FaultConfig
    bus: MessageBus
    clock: Clock
    sensor: ScalarSensor
    commands: Subscription[CommandMsg]

    @staticmethod
    def from_config(
        cfg: PactConfig,
        bus: MessageBus,
        clock: Clock,
        sensor: ScalarSensor,
    ) -> ElectricalApp:
        """Assemble an ElectricalApp and subscribe it to inbound commands.

        Args:
            cfg: Top-level PactConfig (cfg.fault is retained for the limit + heartbeat).
            bus: The MessageBus to publish onto and subscribe to.
            clock: Injected Clock.
            sensor: The ScalarSensor reading power draw in Watts.

        Returns:
            An ElectricalApp holding a fresh CommandMsg subscription.
        """
        return ElectricalApp(
            cfg=cfg.fault,
            bus=bus,
            clock=clock,
            sensor=sensor,
            commands=bus.subscribe(CommandMsg),
        )

    def sample(self) -> None:
        """Read the power draw, publish telemetry, and emit a fault if over the limit.

        On a sensor read error the cycle is skipped (no telemetry, no fault) -- a
        transient read failure surfaces as missing telemetry, which the watchdog/ground
        observe; there is no dedicated sensor-fault code.
        """
        result = self.sensor.read()
        if not isinstance(result, Ok):
            return
        power_w = result.value
        self.bus.publish(
            TelemetryEventMsg(
                msg_type=MessageType.TELEMETRY_EVENT,
                timestamp_utc=self.clock.wall_clock_iso(),
                subsystem=SUBSYSTEM,
                event_name="electrical_sample",
                payload={"power_w": power_w},
            )
        )
        if power_w > self.cfg.power_limit_w:
            self.bus.publish(
                FaultEventMsg(
                    msg_type=MessageType.FAULT_EVENT,
                    timestamp_utc=self.clock.wall_clock_iso(),
                    fault_code=FaultCode.POWER_OVER_LIMIT,
                    subsystem=SUBSYSTEM,
                    detail=(f"power {power_w:.1f}W exceeds limit {self.cfg.power_limit_w:.1f}W"),
                )
            )

    def handle_commands(self) -> None:
        """Acknowledge each pending CommandMsg targeting this subsystem via telemetry."""
        while not self.commands.empty():
            command = self.commands.get_nowait()
            if command.target != SUBSYSTEM:
                continue
            self.bus.publish(
                TelemetryEventMsg(
                    msg_type=MessageType.TELEMETRY_EVENT,
                    timestamp_utc=self.clock.wall_clock_iso(),
                    subsystem=SUBSYSTEM,
                    event_name="command_ack",
                    payload={"command_id": command.command_id, "seq": command.seq},
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
