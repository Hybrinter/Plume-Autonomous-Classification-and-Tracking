"""Fault subsystem app: heartbeat watchdog + fault-to-mode router over the bus.

Subscribes to HeartbeatMsg and FaultEventMsg from every subsystem, runs the pure
watchdog each tick, applies the SAFE-mode policy, and publishes ModeChangeMsg. The
imperative shell owns the bus subscriptions, the clock, and the watchdog-entry dict;
all decision logic is pure (watchdog.check_heartbeats, policy.decide_mode_change).

Contains:
  - FaultApp: frozen holder of config/bus/clock/subscriptions. from_config() subscribes
    to the bus; initial_entries() seeds the watchdog dict; tick() runs one
    drain-heartbeats -> route-faults -> watchdog cycle (threading the entries dict and
    publishing any ModeChangeMsg); run() is the periodic loop.

Non-obvious notes:
  - The arbiter/watchdog interval time is Clock.monotonic_s(); message timestamps use
    Clock.wall_clock_iso(). tick() takes `now` explicitly so it is deterministic in tests.
  - Thermal/power/inference-latency self-checks live in their producing subsystems, not
    here; this app only watches heartbeats and routes already-raised FaultEventMsgs.

Satisfies: REQ-SAFE-HIGH-002, REQ-OPER-HIGH-002.
"""

from __future__ import annotations

# stdlib
import threading
from dataclasses import dataclass, field, replace

# internal
from flight.fault.policy import can_exit_safe, decide_mode_change, exit_safe_mode
from flight.fault.watchdog import WatchdogEntry, build_entries, check_heartbeats
from flight.libs.bus import MessageBus, Subscription
from flight.libs.config import FaultConfig, PactConfig
from flight.libs.messages import (
    CommandAckMsg,
    FaultEventMsg,
    HeartbeatMsg,
    RoutedCommandMsg,
    SafetyStateMsg,
)
from flight.libs.time import Clock
from flight.libs.types import AckStatus, FaultCode, MessageType, SystemMode

_EXIT_SAFE = "EXIT_SAFE"


@dataclass(slots=True)
class SafetyLatch:
    """Mutable SAFE-latch state owned by the fault app shell (the inhibit authority).

    Fields:
        safe_latched: True once a SAFE-triggering fault latched SAFE, until a successful
            EXIT_SAFE clears it.
        safe_reason: The fault code that latched SAFE (NONE when not latched).
    """

    safe_latched: bool = False
    safe_reason: FaultCode = FaultCode.NONE


@dataclass(frozen=True)
class FaultApp:
    """FDIR subsystem app: heartbeat watchdog and fault-to-mode router over the bus.

    Frozen to prevent field reassignment; the held bus/clock/subscriptions are mutable
    services injected by the composition root.
    """

    cfg: FaultConfig
    bus: MessageBus
    clock: Clock
    monitored: tuple[str, ...]
    heartbeats: Subscription[HeartbeatMsg]
    faults: Subscription[FaultEventMsg]
    routed: Subscription[RoutedCommandMsg]
    safety: SafetyLatch = field(default_factory=SafetyLatch)

    @staticmethod
    def from_config(
        cfg: PactConfig,
        bus: MessageBus,
        clock: Clock,
        monitored: tuple[str, ...],
    ) -> FaultApp:
        """Assemble a FaultApp and subscribe it to heartbeats, faults, and routed commands.

        Args:
            cfg: Top-level PactConfig (cfg.fault is retained).
            bus: The MessageBus to subscribe to and publish onto.
            clock: Injected Clock.
            monitored: Names of the subsystems whose heartbeats are watched.

        Returns:
            A FaultApp holding fresh HeartbeatMsg, FaultEventMsg, and RoutedCommandMsg
            subscriptions and a cleared SafetyLatch.
        """
        return FaultApp(
            cfg=cfg.fault,
            bus=bus,
            clock=clock,
            monitored=monitored,
            heartbeats=bus.subscribe(HeartbeatMsg),
            faults=bus.subscribe(FaultEventMsg),
            routed=bus.subscribe(RoutedCommandMsg),
            safety=SafetyLatch(),
        )

    def initial_entries(self) -> dict[str, WatchdogEntry]:
        """Seed the watchdog entries dict for all monitored subsystems at the current time."""
        entries: dict[str, WatchdogEntry] = build_entries(
            self.monitored, self.cfg.watchdog_interval_s, self.clock.monotonic_s()
        )
        return entries

    def tick(self, entries: dict[str, WatchdogEntry], now: float) -> dict[str, WatchdogEntry]:
        """Run one watchdog + fault-routing + safety-state cycle, publishing the outcomes.

        Drains heartbeats (resetting miss counts), routes fault events + WATCHDOG_EXPIRE
        through the SAFE policy (publishing ModeChangeMsg and latching SAFE), handles any
        routed EXIT_SAFE command (gated on no SAFE-triggering fault this tick), then publishes
        the fault-owned SafetyStateMsg (the inhibit authority the command router consumes).

        Args:
            entries: Current watchdog entries (threaded state; not mutated in place).
            now: Current monotonic seconds.

        Returns:
            The updated watchdog entries dict.
        """
        working = dict(entries)
        iso = self.clock.wall_clock_iso()
        safe_faults_this_tick: set[FaultCode] = set()

        while not self.heartbeats.empty():
            heartbeat = self.heartbeats.get_nowait()
            if heartbeat.subsystem in working:
                working[heartbeat.subsystem] = replace(
                    working[heartbeat.subsystem], last_heartbeat_time=now, miss_count=0
                )

        while not self.faults.empty():
            event = self.faults.get_nowait()
            change = decide_mode_change(event, iso)
            if change is not None:
                self.bus.publish(change)
                self.safety.safe_latched = True
                self.safety.safe_reason = event.fault_code
                safe_faults_this_tick.add(event.fault_code)

        updated: dict[str, WatchdogEntry]
        updated, watchdog_faults = check_heartbeats(
            working, now, self.cfg.watchdog_max_miss_count, iso
        )
        for fault in watchdog_faults:
            change = decide_mode_change(fault, iso)
            if change is not None:
                self.bus.publish(change)
                self.safety.safe_latched = True
                self.safety.safe_reason = fault.fault_code
                safe_faults_this_tick.add(fault.fault_code)

        self._handle_exit_safe(bool(safe_faults_this_tick), iso)

        self.bus.publish(
            SafetyStateMsg(
                msg_type=MessageType.SAFETY_STATE,
                timestamp_utc=iso,
                mode=SystemMode.SAFE if self.safety.safe_latched else SystemMode.IDLE,
                active_faults=tuple(sorted(safe_faults_this_tick, key=lambda c: c.value)),
                safe_latched=self.safety.safe_latched,
                safe_reason=self.safety.safe_reason,
            )
        )
        return updated

    def _handle_exit_safe(self, safe_fault_this_tick: bool, iso: str) -> None:
        """Drain routed EXIT_SAFE commands; un-latch SAFE when the triggering fault is cleared.

        Args:
            safe_fault_this_tick: True if any SAFE-triggering fault fired in this tick (the
                "fault not yet cleared" gate). An EXIT_SAFE is refused while this holds.
            iso: Wall-clock ISO timestamp for the produced messages.

        Notes:
            Commands other than EXIT_SAFE that route to the fault app are ignored (the command
            dictionary only targets the fault app with EXIT_SAFE). A successful exit publishes a
            ModeChangeMsg(IDLE) (consumed by the arbiter to un-latch) plus an ACCEPTED exec ack;
            a refused exit publishes a REJECTED exec ack and leaves SAFE latched.
        """
        while not self.routed.empty():
            command = self.routed.get_nowait()
            if command.command_id != _EXIT_SAFE:
                continue
            if can_exit_safe(self.safety.safe_latched, safe_fault_this_tick):
                self.bus.publish(exit_safe_mode(command.source, iso))
                self.safety.safe_latched = False
                self.safety.safe_reason = FaultCode.NONE
                self._publish_exec_ack(command, AckStatus.ACCEPTED, FaultCode.NONE, "safe exited")
            else:
                self._publish_exec_ack(
                    command,
                    AckStatus.REJECTED,
                    FaultCode.COMMAND_INVALID,
                    "cannot exit safe: not latched or a triggering fault is still active",
                )

    def _publish_exec_ack(
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
        """Run the FDIR loop until stop_event is set.

        Seeds the watchdog entries, then ticks every cfg.watchdog_interval_s seconds.
        Uses stop_event.wait(timeout=...) so shutdown is immediate.

        Args:
            stop_event: threading.Event; the loop exits cleanly once it is set.
        """
        entries = self.initial_entries()
        while not stop_event.is_set():
            now = self.clock.monotonic_s()
            entries = self.tick(entries, now)
            stop_event.wait(timeout=self.cfg.watchdog_interval_s)
