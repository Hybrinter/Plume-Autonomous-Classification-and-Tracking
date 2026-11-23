"""Fault subsystem app: heartbeat watchdog + fault-to-request router over the bus.

Subscribes to HeartbeatMsg and FaultEventMsg from every subsystem, runs the pure
watchdog each tick, publishes SystemModeRequestMsg for SAFE-triggering faults,
consumes SystemModeActivatedMsg for authorized recovery, and publishes the
fault-owned SafetyStateMsg evidence each tick (inhibit authority). The app never
executes a mode change itself: the external mode authority owns every
ACCEPTED/DENIED decision and activation record.

Contains:
  - SafetyLatch: mutable latch/evidence counters owned by the shell.
  - FaultApp: frozen holder of config/bus/clock/subscriptions. from_config()
    subscribes to the bus; initial_entries() seeds the watchdog dict; tick() runs
    one drain-heartbeats -> route-faults -> watchdog -> activation -> evidence
    cycle; run() is the periodic loop.

Non-obvious notes:
  - The watchdog interval is Clock.monotonic_s(); message timestamps use
    Clock.wall_clock_iso(). tick() takes `now` explicitly so tests are deterministic.
  - Request identities are shell-uuid4 values generated here; pure policy
    functions take them as explicit arguments.
  - Thermal/power/inference-latency self-checks live in their producing
    subsystems, not here; this app watches heartbeats and routes raised faults.

Satisfies: REQ-SAFE-HIGH-002, REQ-OPER-HIGH-002.
"""

from __future__ import annotations

# stdlib
import threading
import uuid
from dataclasses import dataclass, field, replace

# internal
from flight.fault.policy import (
    decide_mode_request,
    recovery_authorized,
)
from flight.fault.watchdog import WatchdogEntry, build_entries, check_heartbeats
from flight.libs.bus import MessageBus, Subscription
from flight.libs.config import FaultConfig, PactConfig
from flight.libs.messages import (
    FaultEventMsg,
    HeartbeatMsg,
    SafetyStateMsg,
    SystemModeActivatedMsg,
)
from flight.libs.time import Clock
from flight.libs.types import FaultCode, MessageType


@dataclass(slots=True)
class SafetyLatch:
    """Mutable SAFE-latch and evidence counters owned by the fault app shell.

    Fields:
        safe_latched: True once a SAFE-triggering fault latched SAFE, until an
            authorized recovery activation releases it.
        safe_reason: The fault code that latched SAFE (NONE when not latched).
        last_activation: Newest accepted authority activation record, retained
            complete so duplicates/conflicts/stale and negative sequences are
            classified rather than folded into a bare counter.
        evidence_sequence: Per-epoch counter incremented on each SafetyStateMsg.
        released_recovery_request_id: request_id of the consumed recovery
            activation, retained on the release evidence publication; cleared
            when a new SAFE-triggering fault latches.
        consumed_recovery_ids: request ids already spent on a release; a
            replayed record under a consumed id can never re-release.
    """

    safe_latched: bool = False
    safe_reason: FaultCode = FaultCode.NONE
    last_activation: SystemModeActivatedMsg | None = None
    evidence_sequence: int = 0
    released_recovery_request_id: str | None = None
    consumed_recovery_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class FaultApp:
    """FDIR subsystem app: heartbeat watchdog and fault-to-request router.

    Frozen to prevent field reassignment; the held bus/clock/subscriptions are
    mutable services injected by the composition root.
    """

    cfg: FaultConfig
    bus: MessageBus
    clock: Clock
    monitored: tuple[str, ...]
    activation_epoch: str
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
        activation_epoch: str,
    ) -> FaultApp:
        """Assemble a FaultApp and subscribe it to heartbeats, faults, activations.

        Args:
            cfg: Top-level PactConfig (cfg.fault is retained).
            bus: The MessageBus to subscribe to and publish onto.
            clock: Injected Clock.
            monitored: Names of the subsystems whose heartbeats are watched.
            activation_epoch: Composition-root epoch this instance runs under.

        Returns:
            A FaultApp holding fresh subscriptions and a cleared SafetyLatch.
        """
        return FaultApp(
            cfg=cfg.fault,
            bus=bus,
            clock=clock,
            monitored=monitored,
            activation_epoch=activation_epoch,
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
        """Run one watchdog + fault-routing + recovery + evidence cycle.

        Drains heartbeats (resetting miss counts), routes fault events plus
        WATCHDOG_EXPIRE through the SAFE-request policy (publishing
        SystemModeRequestMsg and latching SAFE), consumes activation records
        (tracking sequence and releasing the latch only on an authorized
        recovery), then publishes the fault-owned SafetyStateMsg evidence.

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
            request = decide_mode_request(event, iso, str(uuid.uuid4()))
            if request is not None:
                self.bus.publish(request)
                self.safety.safe_latched = True
                self.safety.safe_reason = event.fault_code
                self.safety.released_recovery_request_id = None
                safe_faults_this_tick.add(event.fault_code)

        updated: dict[str, WatchdogEntry]
        updated, watchdog_faults = check_heartbeats(
            working, now, self.cfg.watchdog_max_miss_count, iso
        )
        for fault in watchdog_faults:
            request = decide_mode_request(fault, iso, str(uuid.uuid4()))
            if request is not None:
                self.bus.publish(request)
                self.safety.safe_latched = True
                self.safety.safe_reason = fault.fault_code
                self.safety.released_recovery_request_id = None
                safe_faults_this_tick.add(fault.fault_code)

        self._consume_activations(bool(safe_faults_this_tick))

        self.safety.evidence_sequence += 1
        self.bus.publish(
            SafetyStateMsg(
                msg_type=MessageType.SAFETY_STATE,
                timestamp_utc=iso,
                active_faults=tuple(sorted(safe_faults_this_tick, key=lambda c: c.value)),
                safe_latched=self.safety.safe_latched,
                safe_reason=self.safety.safe_reason,
                evidence_epoch=self.activation_epoch,
                evidence_sequence=self.safety.evidence_sequence,
                observed_s=now,
                recovery_request_id=self.safety.released_recovery_request_id,
            )
        )
        return updated

    def _consume_activations(self, safe_fault_this_tick: bool) -> None:
        """Drain activation records; release the latch only on authorized recovery.

        The latch holds until an activation is recovery_authorized, carries a
        nonempty request_id, moves SAFE -> INIT under the current epoch with a
        strictly newer sequence, and no SAFE-triggering fault fired this tick.
        A release consumes its request_id; the released evidence then carries
        recovery_request_id so the payload can match it once.
        """
        while not self.activations.empty():
            activation = self.activations.get_nowait()
            if activation.epoch != self.activation_epoch or activation.sequence < 0:
                continue
            last = self.safety.last_activation
            if last is not None and activation.sequence <= last.sequence:
                continue
            if recovery_authorized(
                activation,
                expected_epoch=self.activation_epoch,
                last_sequence=last.sequence if last is not None else None,
                request_id_consumed=bool(
                    activation.request_id
                    and activation.request_id in self.safety.consumed_recovery_ids
                ),
                safe_fault_seen_this_tick=safe_fault_this_tick,
            ):
                self.safety.safe_latched = False
                self.safety.safe_reason = FaultCode.NONE
                self.safety.released_recovery_request_id = activation.request_id
                self.safety.consumed_recovery_ids.add(activation.request_id or "")
            self.safety.last_activation = activation

    def run(self, stop_event: threading.Event) -> None:
        """Run the FDIR loop until stop_event is set.

        Seeds the watchdog entries, then ticks every cfg.watchdog_interval_s
        seconds. Uses stop_event.wait(timeout=...) so shutdown is immediate.
        """
        entries = self.initial_entries()
        while not stop_event.is_set():
            now = self.clock.monotonic_s()
            entries = self.tick(entries, now)
            stop_event.wait(timeout=self.cfg.watchdog_interval_s)
