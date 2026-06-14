"""Mechanical subsystem app: the launch-lock owner + bidirectional gimbal interlock.

The mechanical app owns the LaunchLock HAL device (spec Section 5). Each cycle it publishes the
current LaunchLockStateMsg (+ a telemetry event) and a heartbeat, and it executes a routed
RELEASE_LAUNCH_LOCK command (a hazardous ARM/EXECUTE command already gated by the core router).
It enforces one direction of the bidirectional interlock at actuation: it refuses to release the
pin while the gimbal is being commanded to move (a GimbalCommandMsg seen this cycle), acking the
command REJECTED. The other direction -- the payload inhibiting gimbal motion while the lock is
ENGAGED -- lives in the payload app, driven by the LaunchLockStateMsg this app publishes.

Contains:
  - MechanicalState: mutable holder for the most recent published lock state (telemetry).
  - MechanicalApp: from_config(); tick(); run(); helpers.

Satisfies: REQ-MECH-HIGH-001.
"""

from __future__ import annotations

# stdlib
import threading
from dataclasses import dataclass, field

# internal
from flight.hal.interfaces import LaunchLock
from flight.libs.bus import MessageBus, Subscription
from flight.libs.config import FaultConfig, PactConfig
from flight.libs.messages import (
    CommandAckMsg,
    FaultEventMsg,
    GimbalCommandMsg,
    HeartbeatMsg,
    LaunchLockStateMsg,
    RoutedCommandMsg,
    TelemetryEventMsg,
)
from flight.libs.time import Clock
from flight.libs.types import (
    AckStatus,
    FaultCode,
    GimbalCommandMode,
    LaunchLockState,
    MessageType,
    Ok,
)

SUBSYSTEM = "mechanical"
_RELEASE_LAUNCH_LOCK = "RELEASE_LAUNCH_LOCK"


@dataclass(slots=True)
class MechanicalState:
    """Mutable mechanical-app state.

    Fields:
        last_state: The most recently read launch-lock state (telemetry/inspection).
    """

    last_state: LaunchLockState = LaunchLockState.UNKNOWN


@dataclass(frozen=True)
class MechanicalApp:
    """Mechanical subsystem app: launch-lock owner + release actuation with motion interlock."""

    cfg: FaultConfig
    bus: MessageBus
    clock: Clock
    lock: LaunchLock
    commands: Subscription[RoutedCommandMsg]
    gimbal_cmds: Subscription[GimbalCommandMsg]
    state: MechanicalState = field(default_factory=MechanicalState)

    @staticmethod
    def from_config(
        cfg: PactConfig, bus: MessageBus, clock: Clock, lock: LaunchLock
    ) -> MechanicalApp:
        """Assemble a MechanicalApp subscribing to routed commands + gimbal-command telemetry.

        Args:
            cfg: Top-level PactConfig (cfg.fault is retained for the heartbeat interval).
            bus: The shared MessageBus to publish onto and subscribe from.
            clock: Injected Clock (real or manual).
            lock: The injected LaunchLock driver (SimLaunchLock today; real is deferred).

        Returns:
            A MechanicalApp with fresh RoutedCommandMsg + GimbalCommandMsg subscriptions.
        """
        return MechanicalApp(
            cfg=cfg.fault,
            bus=bus,
            clock=clock,
            lock=lock,
            commands=bus.subscribe(RoutedCommandMsg),
            gimbal_cmds=bus.subscribe(GimbalCommandMsg),
            state=MechanicalState(),
        )

    def tick(self) -> None:
        """Handle a routed release (with the motion interlock), then publish the lock state."""
        moving = self._drain_gimbal_motion()
        while not self.commands.empty():
            command = self.commands.get_nowait()
            if command.target != SUBSYSTEM:
                continue
            if command.command_id == _RELEASE_LAUNCH_LOCK:
                self._handle_release(command, moving)
            else:
                self._publish_ack(
                    command, AckStatus.REJECTED, FaultCode.COMMAND_INVALID, "unsupported command"
                )
        self._publish_lock_state()

    def _drain_gimbal_motion(self) -> bool:
        """Drain gimbal-command telemetry this cycle; return True if any commanded motion seen."""
        moving = False
        while not self.gimbal_cmds.empty():
            cmd = self.gimbal_cmds.get_nowait()
            if cmd.mode in (
                GimbalCommandMode.ABSOLUTE,
                GimbalCommandMode.STOW,
                GimbalCommandMode.HOME,
            ):
                moving = True
            elif cmd.mode is GimbalCommandMode.RATE and (
                cmd.az_value_deg != 0.0 or cmd.el_value_deg != 0.0
            ):
                moving = True
        return moving

    def _handle_release(self, command: RoutedCommandMsg, moving: bool) -> None:
        """Release the lock unless the gimbal is moving; ack the execution outcome."""
        if moving:
            self._publish_ack(
                command,
                AckStatus.REJECTED,
                FaultCode.LAUNCH_LOCK_FAULT,
                "release inhibited: gimbal in motion",
            )
            return
        result = self.lock.release()
        if isinstance(result, Ok):
            self._publish_ack(command, AckStatus.ACCEPTED, FaultCode.NONE, "launch lock released")
        else:
            self.bus.publish(
                FaultEventMsg(
                    msg_type=MessageType.FAULT_EVENT,
                    timestamp_utc=self.clock.wall_clock_iso(),
                    fault_code=FaultCode.LAUNCH_LOCK_FAULT,
                    subsystem=SUBSYSTEM,
                    detail="launch lock release failed",
                )
            )
            self._publish_ack(
                command, AckStatus.REJECTED, FaultCode.LAUNCH_LOCK_FAULT, "release failed"
            )

    def _publish_lock_state(self) -> None:
        """Read the lock state and publish it as a LaunchLockStateMsg + telemetry event."""
        read = self.lock.read_state()
        state = read.value if isinstance(read, Ok) else LaunchLockState.UNKNOWN
        self.state.last_state = state
        self.bus.publish(
            LaunchLockStateMsg(
                msg_type=MessageType.LAUNCH_LOCK_STATE,
                timestamp_utc=self.clock.wall_clock_iso(),
                state=state,
            )
        )
        self.bus.publish(
            TelemetryEventMsg(
                msg_type=MessageType.TELEMETRY_EVENT,
                timestamp_utc=self.clock.wall_clock_iso(),
                subsystem=SUBSYSTEM,
                event_name="launch_lock_state",
                payload={"state": state.value},
            )
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
        """Run the mechanical loop until stop_event is set, emitting periodic heartbeats.

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
