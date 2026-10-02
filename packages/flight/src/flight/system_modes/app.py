"""System-mode authority app: the imperative shell around the pure transition table.

Drains mode requests, routed system-mode commands, fault safety evidence, and sync requests
from the bus. Every request yields one SystemModeTransitionMsg (accepted or denied). Only an
accepted request yields a SystemModeActivatedMsg with the next (epoch, sequence) key. A sync
request replays the current activation unchanged (same key, no new sequence). The first tick
activates SAFE: the system always boots SAFE and waits for an operator command.

Contains:
  - SUBSYSTEM: the routing target and heartbeat name ("system_modes").
  - AuthorityState: the mutable shell state (current activation, counters, evidence).
  - SystemModesApp: from_config() subscribes; tick() runs one drain-decide-publish cycle;
    run() is the periodic loop with heartbeats.

Non-obvious notes:
  - Subsystem requests are decided before ground commands in a tick, so a SAFE request is
    never reordered behind an EXIT_SAFE received in the same tick.
  - Requests are never coalesced. A duplicate request gets its own DENIED transition record.
  - The epoch is supplied by the composition root; the app never reads it from the clock.

Satisfies: REQ-OPER-HIGH-002, REQ-SAFE-EXIT-001.
"""

from __future__ import annotations

# stdlib
import threading
from dataclasses import dataclass, field

# internal
from flight.libs.bus import MessageBus, Subscription
from flight.libs.config import FaultConfig, PactConfig
from flight.libs.messages import (
    ActivationKey,
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
from flight.libs.types import AckStatus, CommandId, FaultCode, MessageType, SystemMode
from flight.system_modes.transitions import (
    Decision,
    ModeRequest,
    RequestKind,
    SafetyEvidence,
    decide,
)

SUBSYSTEM = "system_modes"

_MODES_BY_NAME: dict[str, SystemMode] = {mode.value: mode for mode in SystemMode}


@dataclass(slots=True)
class AuthorityState:
    """Mutable authority state owned by the app shell.

    Fields:
        active: The latest activation (None before the first accepted request).
        sequence: The last allocated activation sequence in this epoch.
        transitions: Count of transition records published (for transition IDs).
        evidence: The latest fault-published safety evidence.
    """

    active: SystemModeActivatedMsg | None = None
    sequence: int = 0
    transitions: int = 0
    evidence: SafetyEvidence = field(default_factory=SafetyEvidence)

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
            cfg: Top-level PactConfig (cfg.fault supplies the loop and heartbeat interval).
            bus: The MessageBus to subscribe to and publish onto.
            clock: Injected Clock (message timestamps only).
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
        while not self.safety.empty():
            msg = self.safety.get_nowait()
            self.state.evidence = SafetyEvidence(msg.safe_latched, msg.active_faults)
        if self.state.active is None:
            self._process(
                ModeRequest(SystemMode.SAFE, RequestKind.SUBSYSTEM), f"{self.epoch}-boot", SUBSYSTEM
            )
        while not self.requests.empty():
            request = self.requests.get_nowait()
            self._process(
                ModeRequest(request.requested_mode, RequestKind.SUBSYSTEM),
                request.request_id,
                request.requested_by,
            )
        while not self.routed.empty():
            command = self.routed.get_nowait()
            if command.target == SUBSYSTEM:
                self._handle_command(command)
        while not self.syncs.empty():
            self.syncs.get_nowait()
            if self.state.active is not None:
                self.bus.publish(self.state.active)

    def _handle_command(self, command: RoutedCommandMsg) -> None:
        """Map a routed system-mode command to a request, decide it, and ACK it."""
        request: ModeRequest | None = None
        if command.command_id == CommandId.EXIT_SAFE.value:
            request = ModeRequest(SystemMode.INIT, RequestKind.EXIT_SAFE)
        elif command.command_id == CommandId.SET_MODE.value:
            target = _MODES_BY_NAME.get(str(command.params.get("mode", "")))
            if target is not None:
                request = ModeRequest(target, RequestKind.SET_MODE)
        if request is None:
            self._ack(
                command, AckStatus.REJECTED, FaultCode.COMMAND_INVALID, "invalid mode command"
            )
            return
        request_id = f"cmd-{command.source}-{command.seq}"
        decision = self._process(request, request_id, command.source)
        if decision.accepted:
            self._ack(command, AckStatus.ACCEPTED, FaultCode.NONE, decision.reason)
        else:
            self._ack(command, AckStatus.REJECTED, FaultCode.COMMAND_INVALID, decision.reason)

    def _process(self, request: ModeRequest, request_id: str, requested_by: str) -> Decision:
        """Decide one request and publish its transition record and any activation."""
        previous = self.state.mode
        decision = decide(previous, request, self.state.evidence)
        iso = self.clock.wall_clock_iso()
        sequence: int | None = None
        if decision.accepted and decision.resulting_mode is not None:
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
                resulting_mode=decision.resulting_mode,
                decision=decision.decision,
                reason=f"{requested_by}: {decision.reason}",
                activation_sequence=sequence,
            )
        )
        if sequence is not None and decision.resulting_mode is not None:
            activation = SystemModeActivatedMsg(
                msg_type=MessageType.SYSTEM_MODE_ACTIVATED,
                timestamp_utc=iso,
                key=ActivationKey(self.epoch, sequence),
                previous_mode=previous,
                active_mode=decision.resulting_mode,
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
