"""ISS interface app: bridges the station data link and the internal message bus.

Pumps inbound station commands onto the bus (receive_command -> publish CommandMsg) and
outbound downlink items from the bus to the station (drain DownlinkItemMsg ->
send_downlink). The station owns the RF/downlink path, so this app is pure transport
glue with no command interpretation -- the core/target apps act on the published
CommandMsg. Link errors are reported as FaultEventMsg on the bus.

Contains:
  - IssIfaceApp: from_config() subscribes to outbound DownlinkItemMsg; pump_uplink()
    republishes inbound commands; pump_downlink() forwards outbound items; tick() does
    both; run() is the periodic loop with a heartbeat.

Satisfies: REQ-OPER-HIGH-002, REQ-COMM-HIGH-001.
"""

from __future__ import annotations

# stdlib
import threading
from dataclasses import dataclass

# internal
from flight.hal.interfaces import StationLink
from flight.libs.bus import MessageBus, Subscription
from flight.libs.config import FaultConfig, PactConfig
from flight.libs.messages import DownlinkItemMsg, FaultEventMsg, HeartbeatMsg
from flight.libs.time import Clock
from flight.libs.types import Err, FaultCode, MessageType, Ok

HEARTBEAT_SUBSYSTEM = "iss_iface"


@dataclass(frozen=True)
class IssIfaceApp:
    """Station <-> bus bridge. Frozen holder of the injected link, bus, clock, and config."""

    cfg: FaultConfig
    link: StationLink
    bus: MessageBus
    clock: Clock
    downlink: Subscription[DownlinkItemMsg]

    @staticmethod
    def from_config(
        cfg: PactConfig,
        bus: MessageBus,
        clock: Clock,
        link: StationLink,
    ) -> IssIfaceApp:
        """Assemble an IssIfaceApp and subscribe it to outbound downlink items.

        Args:
            cfg: Top-level PactConfig (cfg.fault is retained for heartbeat timing).
            bus: The MessageBus to publish onto and subscribe to.
            clock: Injected Clock.
            link: The StationLink driver (sim or real).

        Returns:
            An IssIfaceApp holding a fresh DownlinkItemMsg subscription.
        """
        return IssIfaceApp(
            cfg=cfg.fault,
            link=link,
            bus=bus,
            clock=clock,
            downlink=bus.subscribe(DownlinkItemMsg),
        )

    def pump_uplink(self) -> int:
        """Drain all pending station commands, publishing each onto the bus.

        Returns:
            The number of CommandMsg published. Stops early and emits a FaultEventMsg
            if the link reports an error.
        """
        count = 0
        while True:
            result = self.link.receive_command()
            if isinstance(result, Err):
                self._publish_fault(result.error, "station uplink receive failed")
                break
            command = result.value
            if command is None:
                break
            self.bus.publish(command)
            count += 1
        return count

    def pump_downlink(self) -> int:
        """Drain all pending downlink items from the bus, forwarding each to the station.

        Returns:
            The number of items successfully sent. A send error emits a FaultEventMsg
            and is not counted.
        """
        count = 0
        while not self.downlink.empty():
            item = self.downlink.get_nowait()
            result = self.link.send_downlink(item)
            if isinstance(result, Ok):
                count += 1
            else:
                self._publish_fault(result.error, "station downlink send failed")
        return count

    def tick(self) -> None:
        """Pump inbound commands and outbound downlinks once."""
        self.pump_uplink()
        self.pump_downlink()

    def run(self, stop_event: threading.Event) -> None:
        """Run the bridge loop until stop_event is set, emitting periodic heartbeats.

        Ticks every cfg.watchdog_interval_s and publishes a HeartbeatMsg on the same
        cadence. (A production link would poll faster; the interval is reused here for
        simplicity.)

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
                        subsystem=HEARTBEAT_SUBSYSTEM,
                        sequence=sequence,
                    )
                )
                sequence += 1
                last_heartbeat = now
            stop_event.wait(timeout=self.cfg.watchdog_interval_s)

    def _publish_fault(self, code: FaultCode, detail: str) -> None:
        """Publish a FaultEventMsg from the iss_iface subsystem onto the bus."""
        self.bus.publish(
            FaultEventMsg(
                msg_type=MessageType.FAULT_EVENT,
                timestamp_utc=self.clock.wall_clock_iso(),
                fault_code=code,
                subsystem=HEARTBEAT_SUBSYSTEM,
                detail=detail,
            )
        )
