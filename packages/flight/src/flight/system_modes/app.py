"""System-mode authority app: the imperative shell around the pure transition table.

Drains mode requests, routed system-mode commands, fault safety evidence, and sync requests
from the bus. Every request yields one SystemModeTransitionMsg (accepted or denied). Only an
accepted request yields a SystemModeActivatedMsg with the next (epoch, sequence) pair. A sync
request under the matching epoch replays the current activation unchanged (same pair, no new
sequence, no transition record). The first tick activates SAFE: the system always boots SAFE
and waits for an operator command.

Contains:
  - SUBSYSTEM: the routing target and heartbeat name ("system_modes").
  - AuthorityState: the mutable shell state (current activation, counters, evidence).
  - SystemModesApp: from_config() subscribes; tick() runs one drain-decide-publish cycle;
    run() is the periodic loop with heartbeats.

Non-obvious notes:
  - Subsystem requests are decided before ground commands in a tick, so a SAFE request is
    never reordered behind an EXIT_SAFE received in the same tick; a SAFE request observed in
    a tick -- subsystem or routed command, even a denied repeat -- also denies that tick's
    EXIT_SAFE with a DENIED transition record.
  - A SAFE-triggering fault in any accepted evidence record denies recovery for the whole
    drain, so a newer clear record published in the same batch cannot undo it.
  - Requests are never coalesced. A duplicate request gets its own DENIED transition record.
  - The epoch is supplied by the composition root; the app never reads it from the clock.
  - Safety evidence acceptance mirrors the command router: matching epoch, nonnegative
    strictly increasing evidence_sequence, finite nonfuture observed_s. A non-SAFE
    transition additionally requires fresh evidence at decision time (within the watchdog
    interval); SAFE is always decidable without it. decide() stays pure on the latest
    accepted SafetyEvidence; freshness is a shell check.
  - INIT -> IDLE is the one keyed transition: the shell accepts it only for the payload's
    own verified completion request ("graph_intent:init_complete") carrying the exact
    ActivationKey of the current INIT activation.

Satisfies: REQ-OPER-HIGH-002, REQ-SAFE-EXIT-001.
"""

from __future__ import annotations

# stdlib
import math
import threading
from dataclasses import dataclass, field

# internal
from flight.libs.bus import MessageBus, Subscription
from flight.libs.config import FaultConfig, PactConfig
from flight.libs.messages import (
    CommandAckMsg,
    HeartbeatMsg,
    RoutedCommandMsg,
    SafetyStateMsg,
    SystemModeActivatedMsg,
    SystemModeRequestMsg,
    SystemModeSyncRequestMsg,
    SystemModeTransitionMsg,
)
from flight.libs.time import Clock
from flight.libs.types import (
    AckStatus,
    ActivationKey,
    CommandId,
    FaultCode,
    MessageType,
    ModeTransitionDecision,
    SystemMode,
)
from flight.system_modes.transitions import (
    Decision,
    ModeRequest,
    RequestKind,
    SafetyEvidence,
    decide,
)

SUBSYSTEM = "system_modes"

_MODES_BY_NAME: dict[str, SystemMode] = {mode.value: mode for mode in SystemMode}

# The exact reason the payload stamps on its verified INIT completion request;
# the authority accepts INIT -> IDLE only under this marker plus the current key.
_INIT_COMPLETE_REASON = "graph_intent:init_complete"


@dataclass(slots=True)
class AuthorityState:
    """Mutable authority state owned by the app shell.

    Fields:
        active: The latest activation (None before the first accepted request).
        sequence: The last allocated activation sequence in this epoch.
        transitions: Count of transition records published (for transition IDs).
        evidence: The latest accepted fault-published safety evidence.
        evidence_observed_s: Monotonic time of that evidence (for freshness).
        evidence_sequence: Greatest accepted fault-owned evidence sequence.
    """

    active: SystemModeActivatedMsg | None = None
    sequence: int = 0
    transitions: int = 0
    evidence: SafetyEvidence = field(default_factory=SafetyEvidence)
    evidence_observed_s: float | None = None
    evidence_sequence: int = -1

    @property
    def mode(self) -> SystemMode | None:
        """Return the active system mode, or None before the first activation."""
        return None if self.active is None else self.active.active_mode


@dataclass(frozen=True)
class SystemModesApp:
    """System-mode authority over the bus. Frozen; the held state object is mutable."""

    cfg: FaultConfig
    bus: MessageBus
    clock: Clock
    epoch: str
    requests: Subscription[SystemModeRequestMsg]
    routed: Subscription[RoutedCommandMsg]
    safety: Subscription[SafetyStateMsg]
    syncs: Subscription[SystemModeSyncRequestMsg]
    state: AuthorityState = field(default_factory=AuthorityState)

    @staticmethod
    def from_config(cfg: PactConfig, bus: MessageBus, clock: Clock, epoch: str) -> SystemModesApp:
        """Assemble the authority and subscribe it to its inputs.

        Args:
            cfg: Top-level PactConfig (cfg.fault supplies the loop and heartbeat interval
                and the safety-evidence freshness bound).
            bus: The MessageBus to subscribe to and publish onto.
            clock: Injected Clock (message timestamps and evidence freshness).
            epoch: Composition-root-provided authority session epoch.

        Returns:
            A SystemModesApp with fresh subscriptions; its first tick activates SAFE.
        """
        return SystemModesApp(
            cfg=cfg.fault,
            bus=bus,
            clock=clock,
            epoch=epoch,
            requests=bus.subscribe(SystemModeRequestMsg),
            routed=bus.subscribe(RoutedCommandMsg),
            safety=bus.subscribe(SafetyStateMsg),
            syncs=bus.subscribe(SystemModeSyncRequestMsg),
            state=AuthorityState(),
        )

    def tick(self) -> None:
        """Run one cycle: evidence, boot SAFE, then requests, then commands, then sync replies."""
        now = self.clock.monotonic_s()
        drain_faults: list[FaultCode] = []
        while not self.safety.empty():
            msg = self.safety.get_nowait()
            if (
                msg.evidence_epoch != self.epoch
                or msg.evidence_sequence < 0
                or msg.evidence_sequence <= self.state.evidence_sequence
                or not math.isfinite(msg.observed_s)
                or msg.observed_s > now
            ):
                continue
            self.state.evidence_sequence = msg.evidence_sequence
            self.state.evidence_observed_s = msg.observed_s
            self.state.evidence = SafetyEvidence(msg.safe_latched, msg.active_faults)
            drain_faults.extend(msg.active_faults)
        # Active faults accepted anywhere in this drain block recovery for the
        # whole tick even when a newer clear record follows them; the next
        # drain's fresh clear alone decides eligibility again. The persisted
        # evidence stays the newest accepted record, so an already-accepted
        # prior latch is not re-charged here.
        effective = self.state.evidence
        if drain_faults:
            effective = SafetyEvidence(effective.safe_latched, tuple(drain_faults))
        if self.state.active is None:
            self._process(
                ModeRequest(SystemMode.SAFE, RequestKind.SUBSYSTEM),
                f"{self.epoch}-boot",
                SUBSYSTEM,
                evidence=effective,
            )
        safe_requested = False
        while not self.requests.empty():
            request = self.requests.get_nowait()
            if request.requested_mode is SystemMode.SAFE:
                safe_requested = True
            self._process(
                ModeRequest(request.requested_mode, RequestKind.SUBSYSTEM),
                request.request_id,
                request.requested_by,
                request_message=request,
                evidence=effective,
            )
        routed_batch: list[RoutedCommandMsg] = []
        while not self.routed.empty():
            routed_batch.append(self.routed.get_nowait())
        # A later SET_MODE SAFE in this batch denies EXIT_SAFE before any command is decided.
        for command in routed_batch:
            if command.target != SUBSYSTEM or command.command_id != CommandId.SET_MODE.value:
                continue
            mode_value = command.params.get("mode")
            target = _MODES_BY_NAME.get(mode_value) if isinstance(mode_value, str) else None
            if target is SystemMode.SAFE:
                safe_requested = True
                break
        for command in routed_batch:
            if command.target == SUBSYSTEM:
                safe_requested |= self._handle_command(command, safe_requested, effective)
        while not self.syncs.empty():
            sync = self.syncs.get_nowait()
            if sync.expected_epoch != self.epoch:
                continue
            if self.state.active is not None:
                self.bus.publish(self.state.active)

    def _evidence_fresh(self) -> bool:
        """Return True while the accepted safety evidence is inside the watchdog interval."""
        observed = self.state.evidence_observed_s
        if observed is None:
            return False
        return (
            0.0 <= self.clock.monotonic_s() - observed <= (self.cfg.watchdog_interval_s + 1.0e-12)
        )

    def _shell_deny_reason(
        self, request: ModeRequest, message: SystemModeRequestMsg | None
    ) -> str | None:
        """Return a shell-level deny reason for a request, or None to consult the table.

        Freshness gates every non-SAFE transition (SAFE is always decidable --
        fail safe). The keyed INIT completion contract additionally requires
        the payload's own completion reason and the exact ActivationKey of the
        current INIT activation; wrong-key, wrong-epoch, missing-key, and
        non-payload readiness are denied.
        """
        if request.target is SystemMode.SAFE:
            return None
        if not self._evidence_fresh():
            return "safety evidence stale or missing"
        if (
            self.state.mode is SystemMode.INIT
            and request.target is SystemMode.IDLE
            and request.kind is RequestKind.SUBSYSTEM
            and (
                message is None
                or message.requested_by != "payload"
                or message.reason != _INIT_COMPLETE_REASON
                or message.activation_key != ActivationKey(self.epoch, self.state.sequence)
            )
        ):
            return "INIT completion requires the current payload activation key"
        return None

    def _handle_command(
        self, command: RoutedCommandMsg, safe_requested: bool, evidence: SafetyEvidence
    ) -> bool:
        """Map a routed system-mode command to a request, decide it, and ACK it.

        Returns:
            True when the command carried a SAFE mode request -- recognized
            even when the request itself is denied (for example a repeated
            SAFE while already in SAFE) -- so a later EXIT_SAFE in the same
            tick is still denied.
        """
        request: ModeRequest | None = None
        deny: str | None = None
        forced_deny: str | None = None
        safe_command = False
        if command.command_id == CommandId.EXIT_SAFE.value:
            if command.params.get("phase") != "EXECUTE":
                deny = "EXIT_SAFE requires routed EXECUTE phase"
            else:
                request = ModeRequest(SystemMode.INIT, RequestKind.EXIT_SAFE)
                if safe_requested:
                    forced_deny = "SAFE request decided this tick"
        elif command.command_id == CommandId.SET_MODE.value:
            mode_value = command.params.get("mode")
            target = _MODES_BY_NAME.get(mode_value) if isinstance(mode_value, str) else None
            if target is None:
                deny = "invalid mode command"
            else:
                request = ModeRequest(target, RequestKind.SET_MODE)
                safe_command = target is SystemMode.SAFE
        elif command.command_id == CommandId.GIMBAL_STOW.value:
            request = ModeRequest(SystemMode.STOW, RequestKind.SET_MODE)
        else:
            deny = "invalid mode command"
        if deny is not None or request is None:
            self._ack(
                command,
                AckStatus.REJECTED,
                FaultCode.COMMAND_INVALID,
                deny if deny is not None else "invalid mode command",
            )
            return safe_command
        request_id = f"cmd-{command.source}-{command.seq}"
        decision = self._process(
            request,
            request_id,
            command.source,
            evidence=evidence,
            forced_deny_reason=forced_deny,
        )
        if decision.accepted:
            self._ack(command, AckStatus.ACCEPTED, FaultCode.NONE, decision.reason)
        else:
            self._ack(command, AckStatus.REJECTED, FaultCode.COMMAND_INVALID, decision.reason)
        return safe_command

    def _process(
        self,
        request: ModeRequest,
        request_id: str,
        requested_by: str,
        *,
        request_message: SystemModeRequestMsg | None = None,
        evidence: SafetyEvidence | None = None,
        forced_deny_reason: str | None = None,
    ) -> Decision:
        """Decide one request and publish its transition record and any activation.

        A shell-level deny (freshness, keyed INIT completion) takes precedence;
        `forced_deny_reason` supplies a deny that still records a transition
        (same-tick SAFE precedence) where a NACK alone would hide the decision.
        """
        previous = self.state.mode
        deny = self._shell_deny_reason(request, request_message) or forced_deny_reason
        decision = (
            decide(previous, request, evidence if evidence is not None else self.state.evidence)
            if deny is None
            else Decision(ModeTransitionDecision.DENIED, previous, deny)
        )
        resulting = decision.resulting_mode
        if resulting is None:
            # Unreachable in practice: the boot activation always precedes
            # request processing, so a decision always carries a mode.
            return decision
        iso = self.clock.wall_clock_iso()
        sequence: int | None = None
        if decision.accepted:
            self.state.sequence += 1
            sequence = self.state.sequence
        self.state.transitions += 1
        self.bus.publish(
            SystemModeTransitionMsg(
                msg_type=MessageType.SYSTEM_MODE_TRANSITION,
                timestamp_utc=iso,
                transition_id=f"{self.epoch}-t{self.state.transitions}",
                request_id=request_id,
                epoch=self.epoch,
                previous_mode=previous,
                requested_mode=request.target,
                resulting_mode=resulting,
                decision=decision.decision,
                reason=f"{requested_by}: {decision.reason}",
                activation_sequence=sequence,
            )
        )
        if sequence is not None:
            activation = SystemModeActivatedMsg(
                msg_type=MessageType.SYSTEM_MODE_ACTIVATED,
                timestamp_utc=iso,
                epoch=self.epoch,
                sequence=sequence,
                previous_mode=previous,
                active_mode=resulting,
                reason=decision.reason,
                request_id=request_id,
                recovery_authorized=decision.recovery_authorized,
            )
            self.state.active = activation
            self.bus.publish(activation)
        return decision

    def _ack(
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
        """Run the authority loop until stop_event is set, emitting periodic heartbeats.

        Args:
            stop_event: threading.Event; the loop exits cleanly once it is set.
        """
        sequence = 0
        last_heartbeat = self.clock.monotonic_s()
        while not stop_event.is_set():
            self.tick()
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
