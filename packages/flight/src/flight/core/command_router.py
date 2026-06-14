"""Core command-router service: dispatch ingress-validated CommandMsgs to target apps.

The imperative shell around the pure routing core (flight.core.routing). It subscribes to
CommandMsg (published by iss_iface after ingress validation) and SafetyStateMsg (published by
the fault app, the inhibit authority), runs route_command for each command, and publishes the
resulting RoutedCommandMsg (to the target app), CommandAckMsg (router-level ARM/reject/core
acks), and -- on an unroutable target -- a loud COMMAND_UNROUTABLE FaultEventMsg (never a silent
drop). It owns the bus, clock, and the mutable RouterState (armed-state + latest SAFE-latch);
all decisions are pure. Heartbeats like every persistent-loop app.

Contains:
  - RouterState: mutable armed-state map ((source, command_id) -> arm seconds) + safe_latched.
  - CommandRouter: from_config(); tick() (one drain+route cycle); run() (periodic loop).

Satisfies: REQ-COMM-CMD-001.
"""

from __future__ import annotations

# stdlib
import threading
from dataclasses import dataclass, field, replace

# internal
from flight.core.routing import route_command
from flight.libs.bus import MessageBus, Subscription
from flight.libs.commands import hazardous_command_ids, routable_targets
from flight.libs.config import CommandRouterConfig, FaultConfig, PactConfig
from flight.libs.messages import (
    CommandMsg,
    FaultEventMsg,
    HeartbeatMsg,
    SafetyStateMsg,
)
from flight.libs.time import Clock
from flight.libs.types import FaultCode, MessageType

SUBSYSTEM = "command_router"


@dataclass(slots=True)
class RouterState:
    """Mutable router state owned by the shell (threaded through the pure core each tick).

    Fields:
        armed: Per-(source, command_id) ARM timestamp map (monotonic seconds).
        safe_latched: The latest SAFE-latch flag from the fault app's SafetyStateMsg.
    """

    armed: dict[tuple[str, str], float] = field(default_factory=dict)
    safe_latched: bool = False


@dataclass(frozen=True)
class CommandRouter:
    """Core command-router service: routes CommandMsg -> RoutedCommandMsg / ack / fault.

    Holds the injected bus/clock/config, the CommandMsg + SafetyStateMsg subscriptions, the
    derived routable-target + hazardous-command sets, and the mutable RouterState. Constructed
    via from_config(); run() drives the periodic routing loop.
    """

    cfg: CommandRouterConfig
    fault_cfg: FaultConfig
    bus: MessageBus
    clock: Clock
    routable: frozenset[str]
    hazardous: frozenset[str]
    commands: Subscription[CommandMsg]
    safety: Subscription[SafetyStateMsg]
    state: RouterState

    @staticmethod
    def from_config(cfg: PactConfig, bus: MessageBus, clock: Clock) -> CommandRouter:
        """Assemble a CommandRouter, subscribing to inbound commands + safety state.

        Args:
            cfg: Top-level PactConfig (command_router for arm window; fault for heartbeat).
            bus: The shared MessageBus to publish onto and subscribe from.
            clock: Injected Clock (real or manual).

        Returns:
            A CommandRouter with fresh CommandMsg + SafetyStateMsg subscriptions, the routable
            target set and hazardous-command set derived from the command dictionary, and empty
            router state.
        """
        return CommandRouter(
            cfg=cfg.command_router,
            fault_cfg=cfg.fault,
            bus=bus,
            clock=clock,
            routable=routable_targets(),
            hazardous=hazardous_command_ids(),
            commands=bus.subscribe(CommandMsg),
            safety=bus.subscribe(SafetyStateMsg),
            state=RouterState(),
        )

    def tick(self) -> None:
        """Drain safety state then route every pending command, publishing the outcomes."""
        while not self.safety.empty():
            self.state.safe_latched = self.safety.get_nowait().safe_latched
        while not self.commands.empty():
            command = self.commands.get_nowait()
            now = self.clock.monotonic_s()
            result = route_command(
                command,
                self.routable,
                self.hazardous,
                self.state.safe_latched,
                self.state.armed,
                now,
                self.cfg.arm_window_s,
            )
            self.state.armed = result.new_armed
            if result.routed_command is not None:
                self.bus.publish(
                    replace(result.routed_command, timestamp_utc=self.clock.wall_clock_iso())
                )
            if result.ack is not None:
                self.bus.publish(replace(result.ack, timestamp_utc=self.clock.wall_clock_iso()))
            if result.unroutable_detail is not None:
                self.bus.publish(
                    FaultEventMsg(
                        msg_type=MessageType.FAULT_EVENT,
                        timestamp_utc=self.clock.wall_clock_iso(),
                        fault_code=FaultCode.COMMAND_UNROUTABLE,
                        subsystem=SUBSYSTEM,
                        detail=result.unroutable_detail,
                    )
                )

    def run(self, stop_event: threading.Event) -> None:
        """Run the routing loop until stop_event is set, emitting periodic heartbeats.

        Args:
            stop_event: threading.Event; the loop exits cleanly once it is set.
        """
        sequence = 0
        last_heartbeat = self.clock.monotonic_s()
        while not stop_event.is_set():
            self.tick()
            now = self.clock.monotonic_s()
            if now - last_heartbeat >= self.fault_cfg.watchdog_interval_s:
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
            stop_event.wait(timeout=self.fault_cfg.watchdog_interval_s)
