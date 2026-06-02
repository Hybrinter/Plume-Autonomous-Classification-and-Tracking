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
from dataclasses import dataclass, replace

# internal
from flight.fault.policy import decide_mode_change
from flight.fault.watchdog import WatchdogEntry, build_entries, check_heartbeats
from flight.libs.bus import MessageBus, Subscription
from flight.libs.config import FaultConfig, PactConfig
from flight.libs.messages import FaultEventMsg, HeartbeatMsg
from flight.libs.time import Clock


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

    @staticmethod
    def from_config(
        cfg: PactConfig,
        bus: MessageBus,
        clock: Clock,
        monitored: tuple[str, ...],
    ) -> FaultApp:
        """Assemble a FaultApp and subscribe it to heartbeats and fault events.

        Args:
            cfg: Top-level PactConfig (cfg.fault is retained).
            bus: The MessageBus to subscribe to and publish onto.
            clock: Injected Clock.
            monitored: Names of the subsystems whose heartbeats are watched.

        Returns:
            A FaultApp holding fresh HeartbeatMsg and FaultEventMsg subscriptions.
        """
        return FaultApp(
            cfg=cfg.fault,
            bus=bus,
            clock=clock,
            monitored=monitored,
            heartbeats=bus.subscribe(HeartbeatMsg),
            faults=bus.subscribe(FaultEventMsg),
        )

    def initial_entries(self) -> dict[str, WatchdogEntry]:
        """Seed the watchdog entries dict for all monitored subsystems at the current time."""
        return build_entries(
            self.monitored, self.cfg.watchdog_interval_s, self.clock.monotonic_s()
        )

    def tick(self, entries: dict[str, WatchdogEntry], now: float) -> dict[str, WatchdogEntry]:
        """Run one watchdog + fault-routing cycle, publishing any mode changes.

        Drains all pending heartbeats (resetting each known subsystem's miss count),
        routes all pending fault events through the SAFE-mode policy, then runs the
        watchdog and routes any WATCHDOG_EXPIRE faults. Every resulting ModeChangeMsg
        is published to the bus.

        Args:
            entries: Current watchdog entries (threaded state; not mutated in place).
            now: Current monotonic seconds.

        Returns:
            The updated watchdog entries dict.
        """
        working = dict(entries)

        while not self.heartbeats.empty():
            heartbeat = self.heartbeats.get_nowait()
            if heartbeat.subsystem in working:
                working[heartbeat.subsystem] = replace(
                    working[heartbeat.subsystem], last_heartbeat_time=now, miss_count=0
                )

        while not self.faults.empty():
            event = self.faults.get_nowait()
            change = decide_mode_change(event, self.clock.wall_clock_iso())
            if change is not None:
                self.bus.publish(change)

        updated, watchdog_faults = check_heartbeats(
            working, now, self.cfg.watchdog_max_miss_count, self.clock.wall_clock_iso()
        )
        for fault in watchdog_faults:
            change = decide_mode_change(fault, self.clock.wall_clock_iso())
            if change is not None:
                self.bus.publish(change)

        return updated

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
