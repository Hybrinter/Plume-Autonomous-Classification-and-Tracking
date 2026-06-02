# Phase 7: ISS Interface (iss_iface) Subsystem Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `iss_iface` subsystem — the bridge between the ISS/station and the internal message bus — including the deferred `CommandMsg` envelope, a `StationLink` HAL protocol with sim/real drivers, and an `IssIfaceApp` that pumps inbound commands onto the bus and outbound downlink items to the station.

**Architecture:** Replaces the legacy RF comms subsystem (the station owns the RF/downlink path, so PACT implements no CCSDS/TDRSS budget logic). The exact ISS data-interface wire protocol is a deferred decision, hidden behind the `StationLink` Protocol in `flight.hal.interfaces`. `IssIfaceApp` is a thin bus<->station bridge: `receive_command()` from the link -> publish `CommandMsg` to the bus; drain `DownlinkItemMsg` from the bus -> `send_downlink()` to the link. All decision logic is trivial; this is transport glue. The model-chunk-upload reassembly from the legacy `comms/uplink.py` is a *consumer* of this transport (a future model-deploy concern) and is deliberately out of scope.

**Tech Stack:** Python 3.14, frozen dataclasses, `Protocol`-based HAL, typed `MessageBus`, injected `Clock`, `Result[T, E]`. mypy --strict, ruff (line-length 100), import-linter, pytest.

---

## Context the implementer needs

**Spec basis (`docs/superpowers/specs/2026-05-30-pact-iss-payload-fsw-structure-design.md`):**
- `CommandMsg{target, command_id, params, source, seq}` — ground/station -> `iss_iface` -> `core` -> target app (Section 5).
- `StationLink` is one of the HAL device protocols (Section 6); the exact ISS protocol is an explicitly deferred decision designed behind it (Section 14).
- Large artifacts never go on the bus; downlink carries compact items only.

**Verified existing flight surfaces this phase builds on:**
- `flight.libs.types.enums.MessageType` — add a `COMMAND = "COMMAND"` member (enum string mirrors name, per conventions). Existing members include `DOWNLINK_ITEM`. No test enumerates `MessageType` exhaustively (`test_enums.py` only subset-checks `FaultCode`), so the addition is safe.
- `flight.libs.messages` — `DownlinkItemMsg(msg_type, timestamp_utc, priority: DownlinkPriority, payload_bytes: bytes, crc32: int, item_id: str)` already exists; `TelemetryEventMsg`, `FaultEventMsg`, `HeartbeatMsg`, `utc_now_iso` exist. `TelemetryEventMsg.payload` is typed `dict[str, str | int | float | bool]` — mirror that for `CommandMsg.params`.
- `flight.libs.types` — `Ok`, `Err`, `Result`, `FaultCode` (incl. `COMM_TIMEOUT`), `MessageType`, `DownlinkPriority`.
- `flight.hal.interfaces` — pattern: `@runtime_checkable` `Protocol`, methods returning `Result[..., FaultCode]`; `__init__.py` re-exports the protocols. Sim drivers in `drivers_sim`, real stubs (lazy SDK or plain no-op) in `drivers_real`; each `__init__.py` re-exports.
- `flight.libs.bus` — `MessageBus.subscribe(type[T]) -> Subscription[T]`, `.publish(object)`; `Subscription.empty()`, `.get_nowait()`.
- `flight.libs.time` — `Clock.monotonic_s()`, `.wall_clock_iso()`; `ManualClock`.
- `flight.libs.config` — `PactConfig`, `FaultConfig(watchdog_interval_s=5.0, ...)`.

**Layering (import-linter), must hold:**
- `flight.iss_iface` may import only `flight.hal.interfaces` + `flight.libs.*` (+ its own submodules). It must NOT import `flight.hal.drivers_sim`/`drivers_real` (the `drivers-from-composition-roots-only` contract) — depend only on the `StationLink` protocol.
- `flight.hal.interfaces.station` imports only `flight.libs.*`.
- `drivers_sim.station` and `drivers_real.station` must NOT import each other (`drivers-independent` contracts).
- Tests may import the concrete drivers freely (tests are outside the `flight` package).

**mypy note (carried from Phase 6):** the CI gate `uv run mypy packages` resolves cross-package `flight.*` imports to `Any`, so a function whose declared return is a generic parameterized by an imported type can trip `--strict`'s `no-any-return`. If that happens, assign to a locally-annotated variable first (e.g. `item: DownlinkItemMsg = self.downlink.get_nowait()`) — do NOT add `# type: ignore`. The methods in this plan return `int`/`None`/same-module types, so this is unlikely, but follow this rule if mypy complains.

---

### Task 1: CommandMsg envelope + MessageType.COMMAND

**Files:**
- Modify: `packages/flight/src/flight/libs/types/enums.py` (add `COMMAND` to `MessageType`)
- Modify: `packages/flight/src/flight/libs/messages/messages.py` (add `CommandMsg`)
- Modify: `packages/flight/src/flight/libs/messages/__init__.py` (export `CommandMsg`)
- Test: `packages/flight/tests/test_command_envelope.py`

- [ ] **Step 1: Write the failing test**

Create `packages/flight/tests/test_command_envelope.py`:

```python
"""Tests for the CommandMsg station-command envelope."""

from flight.libs.messages import CommandMsg
from flight.libs.types import MessageType


def test_command_msg_fields() -> None:
    """CommandMsg carries target, command_id, params, source, and seq."""
    cmd = CommandMsg(
        msg_type=MessageType.COMMAND,
        timestamp_utc="2026-06-01T00:00:00.000Z",
        target="payload",
        command_id="set_mode",
        params={"mode": "ACTIVE", "dwell_s": 30},
        source="ground",
        seq=7,
    )
    assert cmd.target == "payload"
    assert cmd.command_id == "set_mode"
    assert cmd.params["mode"] == "ACTIVE"
    assert cmd.seq == 7
    assert cmd.msg_type is MessageType.COMMAND


def test_command_type_value_mirrors_name() -> None:
    """The new MessageType.COMMAND value mirrors its member name."""
    assert MessageType.COMMAND.value == "COMMAND"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/flight/tests/test_command_envelope.py -v`
Expected: FAIL (ImportError: cannot import name `CommandMsg`, or AttributeError on `MessageType.COMMAND`).

- [ ] **Step 3: Add the MessageType.COMMAND member**

In `packages/flight/src/flight/libs/types/enums.py`, in the `MessageType` enum, add the `COMMAND` member immediately after `MODE_CHANGE`:

```python
    MODE_CHANGE = "MODE_CHANGE"
    COMMAND = "COMMAND"
    STORAGE_WRITE = "STORAGE_WRITE"
```

(Insert the `COMMAND = "COMMAND"` line; leave the surrounding members unchanged.)

- [ ] **Step 4: Add the CommandMsg dataclass**

In `packages/flight/src/flight/libs/messages/messages.py`, add this dataclass immediately after the `ModeChangeMsg` definition (before `StorageWriteMsg`):

```python
@dataclass(frozen=True)
class CommandMsg:
    """Ground/station command routed via iss_iface to a target subsystem.

    The standard command envelope: the station/ground sends a CommandMsg to iss_iface,
    which publishes it onto the bus for the core/target app to act on. params holds only
    JSON-serializable primitives. seq is a monotonic per-source counter for ordering and
    de-duplication.
    """

    msg_type: MessageType  # must be MessageType.COMMAND
    timestamp_utc: str  # ISO 8601, millisecond precision
    target: str  # destination subsystem name (e.g. "payload", "fault")
    command_id: str  # command identifier / opcode (e.g. "set_mode")
    params: dict[str, str | int | float | bool]  # serializable command parameters only
    source: str  # command origin (e.g. "ground", "station_ops")
    seq: int  # monotonic per-source command sequence number
```

- [ ] **Step 5: Export CommandMsg**

In `packages/flight/src/flight/libs/messages/__init__.py`, add `CommandMsg` to both the import block and `__all__` (keep alphabetical ordering: it goes after `BlobMeta`):

```python
from flight.libs.messages.messages import (
    BlobMeta,
    CommandMsg,
    DownlinkItemMsg,
    FaultEventMsg,
    GimbalCommandMsg,
    HeartbeatMsg,
    InferenceResultMsg,
    ModeChangeMsg,
    ProcessedFrameMsg,
    RawFrameMsg,
    StorageWriteMsg,
    TelemetryEventMsg,
    UploadChunkMsg,
    utc_now_iso,
)

__all__ = [
    "BlobMeta",
    "CommandMsg",
    "DownlinkItemMsg",
    "FaultEventMsg",
    "GimbalCommandMsg",
    "HeartbeatMsg",
    "InferenceResultMsg",
    "ModeChangeMsg",
    "ProcessedFrameMsg",
    "RawFrameMsg",
    "StorageWriteMsg",
    "TelemetryEventMsg",
    "UploadChunkMsg",
    "utc_now_iso",
]
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest packages/flight/tests/test_command_envelope.py -v`
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add packages/flight/src/flight/libs/types/enums.py packages/flight/src/flight/libs/messages/messages.py packages/flight/src/flight/libs/messages/__init__.py packages/flight/tests/test_command_envelope.py
git commit -m "feat(libs): add CommandMsg station-command envelope + MessageType.COMMAND"
```

---

### Task 2: StationLink HAL protocol + sim/real drivers

**Files:**
- Create: `packages/flight/src/flight/hal/interfaces/station.py`
- Create: `packages/flight/src/flight/hal/drivers_sim/station.py`
- Create: `packages/flight/src/flight/hal/drivers_real/station.py`
- Modify: `packages/flight/src/flight/hal/interfaces/__init__.py`
- Modify: `packages/flight/src/flight/hal/drivers_sim/__init__.py`
- Modify: `packages/flight/src/flight/hal/drivers_real/__init__.py`
- Test: `packages/flight/tests/test_station_link.py`

- [ ] **Step 1: Write the failing test**

Create `packages/flight/tests/test_station_link.py`:

```python
"""Conformance + behavior tests for the StationLink HAL and its drivers."""

from flight.hal.drivers_real import RealStationLink
from flight.hal.drivers_sim import SimStationLink
from flight.hal.interfaces import StationLink
from flight.libs.messages import CommandMsg, DownlinkItemMsg
from flight.libs.types import DownlinkPriority, MessageType, Ok


def _command(seq: int) -> CommandMsg:
    """Build a CommandMsg targeting the payload subsystem."""
    return CommandMsg(
        msg_type=MessageType.COMMAND,
        timestamp_utc="t",
        target="payload",
        command_id="set_mode",
        params={"mode": "ACTIVE"},
        source="ground",
        seq=seq,
    )


def _downlink_item() -> DownlinkItemMsg:
    """Build a minimal DownlinkItemMsg."""
    return DownlinkItemMsg(
        msg_type=MessageType.DOWNLINK_ITEM,
        timestamp_utc="t",
        priority=DownlinkPriority.HEALTH_TELEMETRY,
        payload_bytes=b"hello",
        crc32=0,
        item_id="item-1",
    )


def test_sim_station_link_satisfies_protocol() -> None:
    """SimStationLink conforms to StationLink (typed + runtime)."""
    link: StationLink = SimStationLink([])
    assert isinstance(link, StationLink)


def test_real_station_link_satisfies_protocol() -> None:
    """RealStationLink conforms to StationLink and constructs without hardware."""
    link: StationLink = RealStationLink()
    assert isinstance(link, StationLink)


def test_sim_receives_scripted_commands_in_order_then_none() -> None:
    """receive_command yields each scripted command once, then Ok(None)."""
    link = SimStationLink([_command(1), _command(2)])
    first = link.receive_command()
    second = link.receive_command()
    third = link.receive_command()
    assert isinstance(first, Ok) and first.value is not None and first.value.seq == 1
    assert isinstance(second, Ok) and second.value is not None and second.value.seq == 2
    assert isinstance(third, Ok) and third.value is None


def test_sim_records_downlinked_items() -> None:
    """send_downlink records each item; downlinked exposes them in order."""
    link = SimStationLink([])
    result = link.send_downlink(_downlink_item())
    assert isinstance(result, Ok)
    assert len(link.downlinked) == 1
    assert link.downlinked[0].item_id == "item-1"


def test_real_station_link_stub_is_inert() -> None:
    """RealStationLink stub returns no pending command and accepts downlinks."""
    link = RealStationLink()
    received = link.receive_command()
    assert isinstance(received, Ok) and received.value is None
    assert isinstance(link.send_downlink(_downlink_item()), Ok)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/flight/tests/test_station_link.py -v`
Expected: FAIL (no module `flight.hal.interfaces.station` / cannot import `SimStationLink`).

- [ ] **Step 3: Write the StationLink protocol**

Create `packages/flight/src/flight/hal/interfaces/station.py`:

```python
"""Station data-link hardware abstraction.

Defines the StationLink protocol: the payload's interface to the ISS/station for
inbound commands and outbound downlink. The station owns the RF/downlink path, so this
abstraction is deliberately thin; the exact ISS data-interface wire protocol is a
deferred decision hidden behind this Protocol (real driver TBD; sim driver drives SIL).
"""

from typing import Protocol, runtime_checkable

from flight.libs.messages import CommandMsg, DownlinkItemMsg
from flight.libs.types import FaultCode, Result


@runtime_checkable
class StationLink(Protocol):
    """Hardware abstraction for the ISS/station command + downlink interface."""

    def receive_command(self) -> Result[CommandMsg | None, FaultCode]:
        """Poll for the next inbound command from the station.

        Returns:
            Result[CommandMsg | None, FaultCode]: Ok(command) when one is pending,
            Ok(None) when the inbound queue is empty, Err(FaultCode.COMM_TIMEOUT) on
            a link error.
        """
        ...

    def send_downlink(self, item: DownlinkItemMsg) -> Result[None, FaultCode]:
        """Hand a downlink item to the station for transmission to the ground."""
        ...
```

- [ ] **Step 4: Write the SimStationLink driver**

Create `packages/flight/src/flight/hal/drivers_sim/station.py`:

```python
"""Simulated station link.

Replays a scripted list of inbound commands (one per receive_command() call, then
Ok(None)) and records every downlinked item for inspection. Satisfies StationLink
structurally; used by SIL and tests.
"""

from flight.libs.messages import CommandMsg, DownlinkItemMsg
from flight.libs.types import FaultCode, Ok, Result


class SimStationLink:
    """Station link that replays scripted commands and records downlinks (sim/SIL)."""

    def __init__(self, inbound: list[CommandMsg]) -> None:
        """Initialize with the inbound commands to replay, in order.

        Args:
            inbound: Commands returned one per receive_command() call, in order.
        """
        self._inbound = inbound
        self._index = 0
        self._downlinked: list[DownlinkItemMsg] = []

    def receive_command(self) -> Result[CommandMsg | None, FaultCode]:
        """Return the next scripted command, or Ok(None) once exhausted."""
        if self._index >= len(self._inbound):
            return Ok(None)
        command = self._inbound[self._index]
        self._index += 1
        return Ok(command)

    def send_downlink(self, item: DownlinkItemMsg) -> Result[None, FaultCode]:
        """Record the downlink item and return Ok(None)."""
        self._downlinked.append(item)
        return Ok(None)

    @property
    def downlinked(self) -> tuple[DownlinkItemMsg, ...]:
        """All items passed to send_downlink, in order (test/SIL inspection hook)."""
        return tuple(self._downlinked)
```

- [ ] **Step 5: Write the RealStationLink stub**

Create `packages/flight/src/flight/hal/drivers_real/station.py`:

```python
"""Real ISS/station data-link driver (stub).

The exact station avionics data interface is a deferred design decision, so this stub
satisfies the StationLink protocol with inert no-ops (no pending command; downlinks
accepted) until the interface is defined. Tests and CI use SimStationLink.
"""

from flight.libs.messages import CommandMsg, DownlinkItemMsg
from flight.libs.types import FaultCode, Ok, Result


class RealStationLink:
    """ISS/station data-link driver (stub). Satisfies StationLink; inert until defined."""

    def receive_command(self) -> Result[CommandMsg | None, FaultCode]:
        """Return Ok(None): no command source wired yet (stub)."""
        return Ok(None)

    def send_downlink(self, item: DownlinkItemMsg) -> Result[None, FaultCode]:
        """Accept and drop the downlink item (stub)."""
        return Ok(None)
```

- [ ] **Step 6: Update the three HAL __init__ files**

Overwrite `packages/flight/src/flight/hal/interfaces/__init__.py`:

```python
"""HAL device interfaces (Protocols). Apps depend only on this module, never on
concrete drivers; the composition root injects the implementation.
"""

from flight.hal.interfaces.gimbal import GimbalActuator, GimbalPosition
from flight.hal.interfaces.sensor import ImagingSensor
from flight.hal.interfaces.station import StationLink

__all__ = ["GimbalActuator", "GimbalPosition", "ImagingSensor", "StationLink"]
```

Overwrite `packages/flight/src/flight/hal/drivers_sim/__init__.py`:

```python
"""Simulation HAL drivers. Reachable only from composition roots (sim/SIL)."""

from flight.hal.drivers_sim.gimbal import SimGimbal
from flight.hal.drivers_sim.sensor import SimSensor
from flight.hal.drivers_sim.station import SimStationLink

__all__ = ["SimGimbal", "SimSensor", "SimStationLink"]
```

Overwrite `packages/flight/src/flight/hal/drivers_real/__init__.py`:

```python
"""Flight hardware HAL drivers. Reachable only from composition roots (flight/core).

Importing this module is safe without any hardware SDK; constructing RealSensor
lazily imports PySpin and raises ImportError if it is absent.
"""

from flight.hal.drivers_real.gimbal import RealGimbal
from flight.hal.drivers_real.sensor import RealSensor
from flight.hal.drivers_real.station import RealStationLink

__all__ = ["RealGimbal", "RealSensor", "RealStationLink"]
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `uv run pytest packages/flight/tests/test_station_link.py -v`
Expected: 6 passed.

- [ ] **Step 8: Commit**

```bash
git add packages/flight/src/flight/hal/interfaces/station.py packages/flight/src/flight/hal/drivers_sim/station.py packages/flight/src/flight/hal/drivers_real/station.py packages/flight/src/flight/hal/interfaces/__init__.py packages/flight/src/flight/hal/drivers_sim/__init__.py packages/flight/src/flight/hal/drivers_real/__init__.py packages/flight/tests/test_station_link.py
git commit -m "feat(hal): add StationLink protocol + sim/real station drivers"
```

---

### Task 3: IssIfaceApp bus<->station bridge

**Files:**
- Create: `packages/flight/src/flight/iss_iface/app.py`
- Modify: `packages/flight/src/flight/iss_iface/__init__.py`
- Test: `packages/flight/tests/test_iss_iface_app.py`

- [ ] **Step 1: Write the failing test**

Create `packages/flight/tests/test_iss_iface_app.py`:

```python
"""Integration tests for the iss_iface app (station <-> bus bridge)."""

from flight.hal.drivers_sim import SimStationLink
from flight.iss_iface.app import IssIfaceApp
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.messages import CommandMsg, DownlinkItemMsg
from flight.libs.time import ManualClock
from flight.libs.types import DownlinkPriority, MessageType


def _command(seq: int) -> CommandMsg:
    """Build a CommandMsg targeting the payload subsystem."""
    return CommandMsg(
        msg_type=MessageType.COMMAND,
        timestamp_utc="t",
        target="payload",
        command_id="set_mode",
        params={"mode": "ACTIVE"},
        source="ground",
        seq=seq,
    )


def _downlink_item(item_id: str) -> DownlinkItemMsg:
    """Build a minimal DownlinkItemMsg with the given id."""
    return DownlinkItemMsg(
        msg_type=MessageType.DOWNLINK_ITEM,
        timestamp_utc="t",
        priority=DownlinkPriority.HEALTH_TELEMETRY,
        payload_bytes=b"x",
        crc32=0,
        item_id=item_id,
    )


def _app(inbound: list[CommandMsg]) -> tuple[IssIfaceApp, MessageBus, SimStationLink]:
    """Assemble an IssIfaceApp over a SimStationLink and a fresh bus."""
    bus = MessageBus()
    link = SimStationLink(inbound)
    app = IssIfaceApp.from_config(PactConfig(), bus, ManualClock(), link)
    return app, bus, link


def test_pump_uplink_publishes_commands_to_bus() -> None:
    """Inbound station commands are republished onto the bus as CommandMsg."""
    app, bus, _link = _app([_command(1), _command(2)])
    cmd_sub = bus.subscribe(CommandMsg)
    count = app.pump_uplink()
    assert count == 2
    received = []
    while not cmd_sub.empty():
        received.append(cmd_sub.get_nowait())
    assert [c.seq for c in received] == [1, 2]


def test_pump_downlink_forwards_bus_items_to_station() -> None:
    """DownlinkItemMsg published on the bus is forwarded to the station link."""
    app, bus, link = _app([])
    bus.publish(_downlink_item("a"))
    bus.publish(_downlink_item("b"))
    count = app.pump_downlink()
    assert count == 2
    assert [item.item_id for item in link.downlinked] == ["a", "b"]


def test_tick_pumps_both_directions() -> None:
    """tick() pumps inbound commands and outbound downlinks in one call."""
    app, bus, link = _app([_command(5)])
    cmd_sub = bus.subscribe(CommandMsg)
    bus.publish(_downlink_item("z"))
    app.tick()
    assert not cmd_sub.empty()
    assert cmd_sub.get_nowait().seq == 5
    assert [item.item_id for item in link.downlinked] == ["z"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/flight/tests/test_iss_iface_app.py -v`
Expected: FAIL (no module `flight.iss_iface.app`).

- [ ] **Step 3: Write the implementation**

Create `packages/flight/src/flight/iss_iface/app.py`:

```python
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
```

- [ ] **Step 4: Export the app**

Overwrite `packages/flight/src/flight/iss_iface/__init__.py`:

```python
"""ISS interface subsystem: the station <-> bus command/downlink bridge."""

from flight.iss_iface.app import IssIfaceApp

__all__ = ["IssIfaceApp"]
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest packages/flight/tests/test_iss_iface_app.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add packages/flight/src/flight/iss_iface/app.py packages/flight/src/flight/iss_iface/__init__.py packages/flight/tests/test_iss_iface_app.py
git commit -m "feat(iss_iface): add IssIfaceApp station<->bus bridge"
```

---

### Task 4: Full gate sweep

**Files:** none (verification only).

- [ ] **Step 1: Run every CI gate, scoped to packages/**

```bash
uv run ruff check packages
uv run ruff format --check packages
uv run mypy packages
uv run lint-imports
uv run pytest packages -m "not e2e"
```

Expected:
- `ruff check packages` -> All checks passed!
- `ruff format --check packages` -> all files already formatted. If any new module/test would be reformatted, run `uv run ruff format packages` and commit with `style: ruff-format iss_iface subsystem`.
- `mypy packages` -> Success (now 98 source files: 94 + station interface + sim station + real station + iss_iface app).
- `lint-imports` -> Contracts: 7 kept, 0 broken. (Confirms `flight.iss_iface` imports only `flight.hal.interfaces` + `flight.libs.*`, and the station drivers stay independent.)
- `pytest packages -m "not e2e"` -> 156 passed, 1 skipped (145 + 11 new: 2 command + 6 station + 3 app).

- [ ] **Step 2: Commit any formatting fix (only if Step 1 required one)**

```bash
git add packages/flight/src/flight/iss_iface packages/flight/src/flight/hal packages/flight/tests
git commit -m "style: ruff-format iss_iface subsystem"
```

---

## HARD RULES for the implementer

- Touch ONLY the files named in Tasks 1-3 (the `flight/iss_iface/*`, `flight/hal/**/station.py` + the three HAL `__init__.py`, the two `libs` files + messages `__init__`, and the three new test files).
- Do NOT modify `src/pact/**` (additive migration; `src/pact` stays untouched). Do NOT stage the pre-existing dirty working-tree entries (`src/pact/fault/detector.py`, `tests/**`, `.idea/*`, `.claude/settings.local.json`, `.coverage`, `bash.exe.stackdump`).
- Commits are LOCAL only; do not push.
- Do NOT migrate the legacy `comms/uplink.py` model-chunk reassembly, `comms/ccsds.py`, `comms/scheduler.py`, or `comms/downlink.py` — they are out of scope (the station owns RF; model-upload is a future consumer of this transport).
- PowerShell/Windows: `uv run ...` for all gates; `git -m` single-quoted strings (no here-strings).
- Python 3.14 / PEP 758: `except A, B:` without parens is valid and ruff-format-normalized — never add parens. Use `from __future__ import annotations` for self-referential return annotations (`-> IssIfaceApp`).
- If mypy reports `no-any-return` on a method, assign to a locally-annotated variable first (per the mypy note above) — never add `# type: ignore`. If a gate fails, fix the cause; never weaken a test assertion.

## Self-Review (spec coverage)

- `CommandMsg{target, command_id, params, source, seq}` envelope. ✓ Task 1.
- `StationLink` HAL protocol (deferred ISS protocol behind it) + sim/real drivers. ✓ Task 2.
- iss_iface bridge: inbound command -> bus; bus downlink item -> station. ✓ Task 3.
- Layering preserved (iss_iface depends only on hal.interfaces + libs; drivers independent). ✓ Task 4 `lint-imports`.
- Out-of-scope RF/CCSDS/model-upload explicitly excluded. ✓ HARD RULES.
```
