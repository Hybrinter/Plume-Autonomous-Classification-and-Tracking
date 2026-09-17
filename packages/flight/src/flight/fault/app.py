"""Fault subsystem app: heartbeat watchdog + system mode manager over the bus.

Subscribes to HeartbeatMsg, FaultEventMsg, ModeRequestMsg, and RoutedCommandMsg.
Each tick runs the watchdog, steps the pure mode manager, and publishes
ModeChangeMsg plus SafetyStateMsg. Boots latched SAFE.

Satisfies: REQ-SAFE-HIGH-002, REQ-OPER-HIGH-002, REQ-SAFE-EXIT-001.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace

from flight.fault.mode import (
    ModeEvent,
    ModeEventKind,
    SystemModeState,
    begin_tick,
    initial_mode_state,
    step,
)
from flight.fault.policy import SAFE_TRIGGERING_FAULTS
from flight.fault.watchdog import WatchdogEntry, build_entries, check_heartbeats
from flight.libs.bus import MessageBus, Subscription
from flight.libs.config import FaultConfig, PactConfig
from flight.libs.messages import (
    CommandAckMsg,
    FaultEventMsg,
    HeartbeatMsg,
    ModeRequestMsg,
    RoutedCommandMsg,
    SafetyStateMsg,
)
from flight.libs.time import Clock
from flight.libs.types import AckStatus, FaultCode, MessageType, ModeRequestReason

_COMMAND_EVENTS: dict[str, ModeEventKind] = {
    "ENTER_INIT": ModeEventKind.ENTER_INIT,
    "ENTER_OPERATE": ModeEventKind.ENTER_OPERATE,
    "ENTER_IDLE": ModeEventKind.ENTER_IDLE,
    "ENTER_STOW": ModeEventKind.ENTER_STOW,
}

_REQUEST_EVENTS: dict[ModeRequestReason, ModeEventKind] = {
    ModeRequestReason.HOMING_COMPLETE: ModeEventKind.HOMING_COMPLETE,
    ModeRequestReason.STOW_COMPLETE: ModeEventKind.STOW_COMPLETE,
    ModeRequestReason.HOMING_FAILED: ModeEventKind.HOMING_FAILED,
    ModeRequestReason.MODEL_SUSPEND: ModeEventKind.MODEL_SUSPEND,
    ModeRequestReason.MODEL_RESUME: ModeEventKind.MODEL_RESUME,
}


@dataclass(slots=True)
class ModeShell:
    """Mutable holder for the frozen SystemModeState."""

    state: SystemModeState = field(default_factory=initial_mode_state)
    active_faults: set[FaultCode] = field(default_factory=set)

    @property
    def safe_latched(self) -> bool:
        """Expose the SAFE latch for tests and inhibit consumers."""
        return self.state.safe_latched

    @property
    def safe_reason(self) -> FaultCode:
        """Expose the latch reason."""
        return self.state.safe_reason


@dataclass(frozen=True)
class FaultApp:
    """FDIR subsystem app: heartbeat watchdog and system mode manager over the bus."""

    cfg: FaultConfig
    bus: MessageBus
    clock: Clock
    monitored: tuple[str, ...]
    heartbeats: Subscription[HeartbeatMsg]
    faults: Subscription[FaultEventMsg]
    routed: Subscription[RoutedCommandMsg]
    requests: Subscription[ModeRequestMsg]
    safety: ModeShell = field(default_factory=ModeShell)

    @staticmethod
    def from_config(
        cfg: PactConfig,
        bus: MessageBus,
        clock: Clock,
        monitored: tuple[str, ...],
    ) -> FaultApp:
        """Assemble a FaultApp and subscribe to heartbeats, faults, commands, and requests."""
        return FaultApp(
            cfg=cfg.fault,
            bus=bus,
            clock=clock,
            monitored=monitored,
            heartbeats=bus.subscribe(HeartbeatMsg),
            faults=bus.subscribe(FaultEventMsg),
            routed=bus.subscribe(RoutedCommandMsg),
            requests=bus.subscribe(ModeRequestMsg),
            safety=ModeShell(),
        )

    def initial_entries(self) -> dict[str, WatchdogEntry]:
        """Seed the watchdog entries dict for all monitored subsystems at the current time."""
        return build_entries(self.monitored, self.cfg.watchdog_interval_s, self.clock.monotonic_s())

    def _apply(self, event: ModeEvent, iso: str) -> None:
        """Step the mode manager and publish ModeChangeMsg when an edge fires."""
        new_state, change = step(self.safety.state, event, iso)
        self.safety.state = new_state
        if change is not None:
            self.bus.publish(change)

    def tick(self, entries: dict[str, WatchdogEntry], now: float) -> dict[str, WatchdogEntry]:
        """Run one watchdog + mode-manager cycle, publishing the outcomes."""
        working = dict(entries)
        iso = self.clock.wall_clock_iso()
        self.safety.state = begin_tick(self.safety.state)
        self.safety.active_faults.clear()

        while not self.heartbeats.empty():
            heartbeat = self.heartbeats.get_nowait()
            if heartbeat.subsystem in working:
                working[heartbeat.subsystem] = replace(
                    working[heartbeat.subsystem], last_heartbeat_time=now, miss_count=0
                )

        while not self.faults.empty():
            event = self.faults.get_nowait()
            if event.fault_code in SAFE_TRIGGERING_FAULTS:
                self.safety.active_faults.add(event.fault_code)
            self._apply(
                ModeEvent(
                    ModeEventKind.FAULT,
                    requested_by=f"safe_mode_entry:{event.fault_code.value}",
                    fault_code=event.fault_code,
                ),
                iso,
            )

        updated, watchdog_faults = check_heartbeats(
            working, now, self.cfg.watchdog_max_miss_count, iso
        )
        for fault in watchdog_faults:
            if fault.fault_code in SAFE_TRIGGERING_FAULTS:
                self.safety.active_faults.add(fault.fault_code)
            self._apply(
                ModeEvent(
                    ModeEventKind.FAULT,
                    requested_by=f"safe_mode_entry:{fault.fault_code.value}",
                    fault_code=fault.fault_code,
                ),
                iso,
            )

        self._handle_mode_requests(iso)
        self._handle_mode_commands(iso)

        self.bus.publish(
            SafetyStateMsg(
                msg_type=MessageType.SAFETY_STATE,
                timestamp_utc=iso,
                mode=self.safety.state.mode,
                active_faults=tuple(sorted(self.safety.active_faults, key=lambda c: c.value)),
                safe_latched=self.safety.state.safe_latched,
                safe_reason=self.safety.state.safe_reason,
            )
        )
        return updated

    def _handle_mode_requests(self, iso: str) -> None:
        """Drain ModeRequestMsg from payload and model deploy."""
        while not self.requests.empty():
            request = self.requests.get_nowait()
            kind = _REQUEST_EVENTS.get(request.reason)
            if kind is None:
                continue
            fault = (
                FaultCode.GIMBAL_RUNAWAY if kind is ModeEventKind.HOMING_FAILED else FaultCode.NONE
            )
            self._apply(
                ModeEvent(
                    kind,
                    requested_by=f"{request.subsystem}:{request.reason.value}",
                    fault_code=fault,
                ),
                iso,
            )

    def _handle_mode_commands(self, iso: str) -> None:
        """Drain routed mode commands and ack accept or reject."""
        while not self.routed.empty():
            command = self.routed.get_nowait()
            kind = _COMMAND_EVENTS.get(command.command_id)
            if kind is None:
                continue
            prior = self.safety.state
            self._apply(
                ModeEvent(
                    kind, requested_by=f"command:{command.source}", fault_code=FaultCode.NONE
                ),
                iso,
            )
            if self.safety.state is prior:
                self._publish_exec_ack(
                    command,
                    AckStatus.REJECTED,
                    FaultCode.COMMAND_INVALID,
                    "mode command refused: illegal edge or latch still active",
                )
            else:
                self._publish_exec_ack(
                    command,
                    AckStatus.ACCEPTED,
                    FaultCode.NONE,
                    f"mode now {self.safety.state.mode.value}",
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
        """Run the FDIR loop until stop_event is set."""
        entries = self.initial_entries()
        while not stop_event.is_set():
            now = self.clock.monotonic_s()
            entries = self.tick(entries, now)
            stop_event.wait(timeout=self.cfg.watchdog_interval_s)
