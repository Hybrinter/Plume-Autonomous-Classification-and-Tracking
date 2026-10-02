"""Fault subsystem app: heartbeat watchdog + fault-to-mode router over the bus.

Subscribes to HeartbeatMsg and FaultEventMsg from every subsystem, runs the pure
watchdog each tick, applies the SAFE-mode policy, publishes ModeChangeMsg, and requests SAFE
from the system-mode authority. The SAFE latch is released only by a recovery-authorized
SystemModeActivatedMsg (the authority owns EXIT_SAFE). The
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
from flight.fault.policy import (
    can_exit_safe,
    decide_mode_change,
    exit_safe_mode,
    safe_mode_request,
)
from flight.fault.watchdog import WatchdogEntry, build_entries, check_heartbeats
from flight.libs.bus import MessageBus, Subscription
from flight.libs.config import FaultConfig, PactConfig
from flight.libs.messages import (
    ActivationKey,
    FaultEventMsg,
    HeartbeatMsg,
    SafetyStateMsg,
    SystemModeActivatedMsg,
)
from flight.libs.time import Clock
from flight.libs.types import FaultCode, MessageType, SystemMode


@dataclass(slots=True)
class SafetyLatch:
    """Mutable SAFE-latch state owned by the fault app shell (the inhibit authority).

    Fields:
        safe_latched: True once a SAFE-triggering fault latched SAFE, until a
            recovery-authorized system-mode activation clears it.
        safe_reason: The fault code that latched SAFE (NONE when not latched).
        authority_mode: The latest system mode the authority activated (None before any).
        last_activation: The key of the latest applied activation (None before any).
        requests_sent: Count of SAFE requests sent (for request IDs).
    """

    safe_latched: bool = False
    safe_reason: FaultCode = FaultCode.NONE
    authority_mode: SystemMode | None = None
    last_activation: ActivationKey | None = None
    requests_sent: int = 0


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
    activations: Subscription[SystemModeActivatedMsg]
    safety: SafetyLatch = field(default_factory=SafetyLatch)

    @staticmethod
    def from_config(
        cfg: PactConfig,
        bus: MessageBus,
        clock: Clock,
        monitored: tuple[str, ...],
    ) -> FaultApp:
        """Assemble a FaultApp and subscribe it to heartbeats, faults, and mode activations.

        Args:
            cfg: Top-level PactConfig (cfg.fault is retained).
            bus: The MessageBus to subscribe to and publish onto.
            clock: Injected Clock.
            monitored: Names of the subsystems whose heartbeats are watched.

        Returns:
            A FaultApp holding fresh HeartbeatMsg, FaultEventMsg, and SystemModeActivatedMsg
            subscriptions and a cleared SafetyLatch.
        """
        return FaultApp(
            cfg=cfg.fault,
            bus=bus,
            clock=clock,
            monitored=monitored,
            heartbeats=bus.subscribe(HeartbeatMsg),
            faults=bus.subscribe(FaultEventMsg),
            activations=bus.subscribe(SystemModeActivatedMsg),
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

        Drains system-mode activations, drains heartbeats (resetting miss counts), routes fault
        events + WATCHDOG_EXPIRE through the SAFE policy (latching SAFE, publishing
        ModeChangeMsg, and requesting SAFE from the authority), releases the latch on a
        recovery-authorized activation when no SAFE-triggering fault fired this tick, then
        publishes the fault-owned SafetyStateMsg.

        Args:
            entries: Current watchdog entries (threaded state; not mutated in place).
            now: Current monotonic seconds.

        Returns:
            The updated watchdog entries dict.
        """
        working = dict(entries)
        iso = self.clock.wall_clock_iso()
        safe_faults_this_tick: set[FaultCode] = set()
        recovery = self._drain_activations()

        while not self.heartbeats.empty():
            heartbeat = self.heartbeats.get_nowait()
            if heartbeat.subsystem in working:
                working[heartbeat.subsystem] = replace(
                    working[heartbeat.subsystem], last_heartbeat_time=now, miss_count=0
                )

        while not self.faults.empty():
            self._route_fault(self.faults.get_nowait(), safe_faults_this_tick, iso)

        updated: dict[str, WatchdogEntry]
        updated, watchdog_faults = check_heartbeats(
            working, now, self.cfg.watchdog_max_miss_count, iso
        )
        for fault in watchdog_faults:
            self._route_fault(fault, safe_faults_this_tick, iso)

        if recovery is not None and can_exit_safe(
            self.safety.safe_latched, bool(safe_faults_this_tick)
        ):
            self.bus.publish(exit_safe_mode(recovery.request_id, iso))
            self.safety.safe_latched = False
            self.safety.safe_reason = FaultCode.NONE

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

    def _route_fault(self, event: FaultEventMsg, safe_faults: set[FaultCode], iso: str) -> None:
        """Latch SAFE for a SAFE-triggering fault and request SAFE from the authority.

        Args:
            event: The fault event (bus-delivered or watchdog-raised).
            safe_faults: The SAFE-triggering faults seen this tick (updated in place).
            iso: Wall-clock ISO timestamp for the produced messages.

        Notes:
            Containment does not wait for the authority: the latch is set immediately. One SAFE
            request is sent per tick, and only while the authority has not activated SAFE.
        """
        change = decide_mode_change(event, iso)
        if change is None:
            return
        self.bus.publish(change)
        if not safe_faults and self.safety.authority_mode is not SystemMode.SAFE:
            self.safety.requests_sent += 1
            self.bus.publish(
                safe_mode_request(event.fault_code, f"fault-{self.safety.requests_sent}", iso)
            )
        self.safety.safe_latched = True
        self.safety.safe_reason = event.fault_code
        safe_faults.add(event.fault_code)

    def _drain_activations(self) -> SystemModeActivatedMsg | None:
        """Drain authority activations; return the newest recovery-authorized one, if any.

        Activations whose key is not newer than the last applied key are ignored, so a
        snapshot replay of an old EXIT_SAFE activation can never release a newer latch.
        """
        recovery: SystemModeActivatedMsg | None = None
        while not self.activations.empty():
            activation = self.activations.get_nowait()
            last = self.safety.last_activation
            if (
                last is not None
                and last.epoch == activation.key.epoch
                and activation.key.sequence <= last.sequence
            ):
                continue
            self.safety.last_activation = activation.key
            self.safety.authority_mode = activation.active_mode
            recovery = activation if activation.recovery_authorized else None
        return recovery

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
