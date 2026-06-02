# Phase 8: Housekeeping Subsystems (Thermal + Electrical) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `thermal` and `electrical` housekeeping subsystems as minimal apps — each emits heartbeats and telemetry, is commandable (no-op acknowledge), and self-reports its threshold fault (`THERMAL_OVER_LIMIT` / `POWER_OVER_LIMIT`) — proving the subsystem topology end-to-end and giving the FDIR policy real fault producers.

**Architecture:** Both subsystems read a scalar housekeeping value behind one shared `ScalarSensor` HAL protocol (a temperature monitor and a power monitor are both scalar sensors). Each app is a thin shell: `sample()` reads the sensor, publishes a `TelemetryEventMsg`, and publishes a threshold `FaultEventMsg` when the reading exceeds its configured limit; `handle_commands()` drains `CommandMsg` targeting the subsystem and acknowledges via telemetry; `run()` loops both plus a heartbeat. All decision logic is trivial and time is injected via `Clock`.

**Tech Stack:** Python 3.14, frozen dataclasses, `Protocol`-based HAL, typed `MessageBus`, injected `Clock`, `Result[T, E]`. mypy --strict, ruff (line-length 100), import-linter, pytest.

---

## Context the implementer needs

**Spec basis:** Section 14 of the design spec lists `thermal`/`electrical`/`mechanical` as peer subsystem apps and calls for "minimal apps (heartbeat + telemetry + commandable no-op) so the topology is provable end-to-end." `mechanical` is intentionally NOT built in this phase (no concrete device yet); its package stays an empty scaffold.

**Why both fault codes already have handlers:** Phase 6 deliberately deferred `check_thermal`/`check_power` to these subsystems. The FDIR policy (`flight.fault.policy.SAFE_TRIGGERING_FAULTS`) already maps `THERMAL_OVER_LIMIT` and `POWER_OVER_LIMIT` to SAFE, so the faults these apps emit are routed correctly with no fault-subsystem change.

**Verified existing flight surfaces:**
- `flight.libs.config.FaultConfig` — `thermal_limit_c: float = 80.0`, `power_limit_w: float = 55.0`, `watchdog_interval_s: float = 5.0`. `PactConfig().fault` is a `FaultConfig`.
- `flight.libs.messages` — `TelemetryEventMsg(msg_type, timestamp_utc, subsystem, event_name, payload: dict[str, str|int|float|bool])`; `FaultEventMsg(msg_type, timestamp_utc, fault_code, subsystem, detail)`; `HeartbeatMsg(msg_type, timestamp_utc, subsystem, sequence)`; `CommandMsg(msg_type, timestamp_utc, target, command_id, params, source, seq)`.
- `flight.libs.types` — `Ok`, `Err`, `Result`, `FaultCode` (incl. `THERMAL_OVER_LIMIT`, `POWER_OVER_LIMIT`), `MessageType` (incl. `TELEMETRY_EVENT`, `FAULT_EVENT`, `HEARTBEAT`).
- `flight.libs.bus` — `MessageBus.subscribe(type[T]) -> Subscription[T]`, `.publish(object)`; `Subscription.empty()`, `.get_nowait()`.
- `flight.libs.time` — `Clock.monotonic_s()`, `.wall_clock_iso()`; `ManualClock`.
- HAL conventions: `@runtime_checkable` Protocol in `flight.hal.interfaces`; sim driver in `drivers_sim`; real stub in `drivers_real` (return a safe nominal value, like `RealGimbal`); each `__init__.py` re-exports.

**Layering (import-linter), must hold:**
- `flight.thermal` and `flight.electrical` may import only `flight.hal.interfaces` + `flight.libs.*`. They must NOT import each other (peer subsystems), `flight.core`, or `flight.hal.drivers_*`.
- `flight.hal.interfaces.scalar` imports only `flight.libs.*`.
- `drivers_sim.scalar` and `drivers_real.scalar` must not import each other.
- Tests may import concrete drivers freely.

**mypy note (carried from Phase 6):** `uv run mypy packages` resolves cross-package `flight.*` imports to `Any`; if a method declared to return a generic parameterized by an imported type trips `no-any-return`, assign to a locally-annotated variable first — never add `# type: ignore`. The methods here return `float`/`None`/`bool`/same-module types, so this is unlikely.

---

### Task 1: ScalarSensor HAL protocol + sim/real drivers

**Files:**
- Create: `packages/flight/src/flight/hal/interfaces/scalar.py`
- Create: `packages/flight/src/flight/hal/drivers_sim/scalar.py`
- Create: `packages/flight/src/flight/hal/drivers_real/scalar.py`
- Modify: `packages/flight/src/flight/hal/interfaces/__init__.py`
- Modify: `packages/flight/src/flight/hal/drivers_sim/__init__.py`
- Modify: `packages/flight/src/flight/hal/drivers_real/__init__.py`
- Test: `packages/flight/tests/test_scalar_sensor.py`

- [ ] **Step 1: Write the failing test**

Create `packages/flight/tests/test_scalar_sensor.py`:

```python
"""Conformance + behavior tests for the ScalarSensor HAL and its drivers."""

from flight.hal.drivers_real import RealScalarSensor
from flight.hal.drivers_sim import SimScalarSensor
from flight.hal.interfaces import ScalarSensor
from flight.libs.types import Ok


def test_sim_scalar_sensor_satisfies_protocol() -> None:
    """SimScalarSensor conforms to ScalarSensor (typed + runtime)."""
    sensor: ScalarSensor = SimScalarSensor([1.0])
    assert isinstance(sensor, ScalarSensor)


def test_real_scalar_sensor_satisfies_protocol() -> None:
    """RealScalarSensor conforms to ScalarSensor and constructs without hardware."""
    sensor: ScalarSensor = RealScalarSensor()
    assert isinstance(sensor, ScalarSensor)


def test_sim_replays_readings_then_holds_last() -> None:
    """read() yields each scripted reading once, then holds the final value."""
    sensor = SimScalarSensor([10.0, 20.0])
    first = sensor.read()
    second = sensor.read()
    third = sensor.read()
    assert isinstance(first, Ok) and first.value == 10.0
    assert isinstance(second, Ok) and second.value == 20.0
    assert isinstance(third, Ok) and third.value == 20.0  # holds last


def test_real_scalar_sensor_reads_nominal_zero() -> None:
    """RealScalarSensor stub returns a safe nominal reading of 0.0."""
    sensor = RealScalarSensor()
    result = sensor.read()
    assert isinstance(result, Ok)
    assert result.value == 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/flight/tests/test_scalar_sensor.py -v`
Expected: FAIL (no module `flight.hal.interfaces.scalar`).

- [ ] **Step 3: Write the ScalarSensor protocol**

Create `packages/flight/src/flight/hal/interfaces/scalar.py`:

```python
"""Scalar housekeeping-sensor hardware abstraction.

Defines the ScalarSensor protocol: a single float reading (e.g. a temperature in
Celsius or a power draw in Watts) sampled from a housekeeping monitor. Shared by the
thermal and electrical subsystems; the meaning and units of the reading are owned by
the consuming subsystem, not the sensor.
"""

from typing import Protocol, runtime_checkable

from flight.libs.types import FaultCode, Result


@runtime_checkable
class ScalarSensor(Protocol):
    """Hardware abstraction for a single-value housekeeping sensor."""

    def read(self) -> Result[float, FaultCode]:
        """Sample the current scalar reading.

        Returns:
            Result[float, FaultCode]: Ok(value) on success; Err(code) on a read error.
        """
        ...
```

- [ ] **Step 4: Write the SimScalarSensor driver**

Create `packages/flight/src/flight/hal/drivers_sim/scalar.py`:

```python
"""Simulated scalar housekeeping sensor.

Replays a fixed list of readings in order, holding the final value once exhausted (a
real housekeeping sensor always reads something). Satisfies ScalarSensor structurally.
"""

from flight.libs.types import FaultCode, Ok, Result


class SimScalarSensor:
    """Scalar sensor that replays scripted readings, holding the last (sim/SIL driver)."""

    def __init__(self, readings: list[float]) -> None:
        """Initialize with the readings to replay, in order.

        Args:
            readings: Non-empty list of readings; read() holds the last once exhausted.
        """
        self._readings = readings
        self._index = 0

    def read(self) -> Result[float, FaultCode]:
        """Return the next reading, holding the final value once exhausted."""
        index = min(self._index, len(self._readings) - 1)
        self._index += 1
        return Ok(self._readings[index])
```

- [ ] **Step 5: Write the RealScalarSensor stub**

Create `packages/flight/src/flight/hal/drivers_real/scalar.py`:

```python
"""Real scalar housekeeping sensor driver (stub).

Returns a safe nominal reading (0.0) until the flight housekeeping bus interface is
wired. Satisfies ScalarSensor; tests and CI use SimScalarSensor.
"""

from flight.libs.types import FaultCode, Ok, Result


class RealScalarSensor:
    """Housekeeping scalar sensor driver (stub). Satisfies ScalarSensor; reads 0.0."""

    def read(self) -> Result[float, FaultCode]:
        """Return a nominal 0.0 reading (stub pending hardware integration)."""
        return Ok(0.0)
```

- [ ] **Step 6: Update the three HAL __init__ files**

Overwrite `packages/flight/src/flight/hal/interfaces/__init__.py`:

```python
"""HAL device interfaces (Protocols). Apps depend only on this module, never on
concrete drivers; the composition root injects the implementation.
"""

from flight.hal.interfaces.gimbal import GimbalActuator, GimbalPosition
from flight.hal.interfaces.scalar import ScalarSensor
from flight.hal.interfaces.sensor import ImagingSensor
from flight.hal.interfaces.station import StationLink

__all__ = ["GimbalActuator", "GimbalPosition", "ImagingSensor", "ScalarSensor", "StationLink"]
```

Overwrite `packages/flight/src/flight/hal/drivers_sim/__init__.py`:

```python
"""Simulation HAL drivers. Reachable only from composition roots (sim/SIL)."""

from flight.hal.drivers_sim.gimbal import SimGimbal
from flight.hal.drivers_sim.scalar import SimScalarSensor
from flight.hal.drivers_sim.sensor import SimSensor
from flight.hal.drivers_sim.station import SimStationLink

__all__ = ["SimGimbal", "SimScalarSensor", "SimSensor", "SimStationLink"]
```

Overwrite `packages/flight/src/flight/hal/drivers_real/__init__.py`:

```python
"""Flight hardware HAL drivers. Reachable only from composition roots (flight/core).

Importing this module is safe without any hardware SDK; constructing RealSensor
lazily imports PySpin and raises ImportError if it is absent.
"""

from flight.hal.drivers_real.gimbal import RealGimbal
from flight.hal.drivers_real.scalar import RealScalarSensor
from flight.hal.drivers_real.sensor import RealSensor
from flight.hal.drivers_real.station import RealStationLink

__all__ = ["RealGimbal", "RealScalarSensor", "RealSensor", "RealStationLink"]
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `uv run pytest packages/flight/tests/test_scalar_sensor.py -v`
Expected: 4 passed.

- [ ] **Step 8: Commit**

```bash
git add packages/flight/src/flight/hal/interfaces/scalar.py packages/flight/src/flight/hal/drivers_sim/scalar.py packages/flight/src/flight/hal/drivers_real/scalar.py packages/flight/src/flight/hal/interfaces/__init__.py packages/flight/src/flight/hal/drivers_sim/__init__.py packages/flight/src/flight/hal/drivers_real/__init__.py packages/flight/tests/test_scalar_sensor.py
git commit -m "feat(hal): add ScalarSensor protocol + sim/real drivers"
```

---

### Task 2: ThermalApp

**Files:**
- Create: `packages/flight/src/flight/thermal/app.py`
- Modify: `packages/flight/src/flight/thermal/__init__.py`
- Test: `packages/flight/tests/test_thermal_app.py`

- [ ] **Step 1: Write the failing test**

Create `packages/flight/tests/test_thermal_app.py`:

```python
"""Tests for the thermal housekeeping app (telemetry + threshold fault + command ack)."""

from flight.hal.drivers_sim import SimScalarSensor
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.messages import CommandMsg, FaultEventMsg, TelemetryEventMsg
from flight.libs.time import ManualClock
from flight.libs.types import FaultCode, MessageType
from flight.thermal.app import ThermalApp


def _app(readings: list[float]) -> tuple[ThermalApp, MessageBus]:
    """Assemble a ThermalApp over a scripted temperature sensor and a fresh bus."""
    bus = MessageBus()
    sensor = SimScalarSensor(readings)
    app = ThermalApp.from_config(PactConfig(), bus, ManualClock(), sensor)
    return app, bus


def test_nominal_reading_publishes_telemetry_no_fault() -> None:
    """A temperature below the limit publishes telemetry and no fault."""
    app, bus = _app([25.0])
    telem = bus.subscribe(TelemetryEventMsg)
    fault = bus.subscribe(FaultEventMsg)
    app.sample()
    assert not telem.empty()
    event = telem.get_nowait()
    assert event.subsystem == "thermal"
    assert event.payload["temperature_c"] == 25.0
    assert fault.empty()


def test_over_limit_reading_publishes_thermal_fault() -> None:
    """A temperature above thermal_limit_c (80.0) publishes THERMAL_OVER_LIMIT."""
    app, bus = _app([95.0])
    fault = bus.subscribe(FaultEventMsg)
    app.sample()
    assert not fault.empty()
    event = fault.get_nowait()
    assert event.fault_code is FaultCode.THERMAL_OVER_LIMIT
    assert event.subsystem == "thermal"


def test_command_targeting_thermal_is_acknowledged() -> None:
    """A CommandMsg targeting 'thermal' produces a command_ack telemetry event."""
    app, bus = _app([25.0])
    telem = bus.subscribe(TelemetryEventMsg)
    bus.publish(
        CommandMsg(
            msg_type=MessageType.COMMAND,
            timestamp_utc="t",
            target="thermal",
            command_id="ping",
            params={},
            source="ground",
            seq=1,
        )
    )
    app.handle_commands()
    assert not telem.empty()
    ack = telem.get_nowait()
    assert ack.event_name == "command_ack"
    assert ack.payload["command_id"] == "ping"


def test_command_for_other_subsystem_ignored() -> None:
    """A CommandMsg targeting another subsystem produces no ack."""
    app, bus = _app([25.0])
    telem = bus.subscribe(TelemetryEventMsg)
    bus.publish(
        CommandMsg(
            msg_type=MessageType.COMMAND,
            timestamp_utc="t",
            target="payload",
            command_id="ping",
            params={},
            source="ground",
            seq=1,
        )
    )
    app.handle_commands()
    assert telem.empty()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/flight/tests/test_thermal_app.py -v`
Expected: FAIL (no module `flight.thermal.app`).

- [ ] **Step 3: Write the implementation**

Create `packages/flight/src/flight/thermal/app.py`:

```python
"""Thermal housekeeping app: temperature telemetry, over-limit fault, command ack.

Minimal subsystem app proving the thermal node is in the topology: each cycle it
samples a temperature via a ScalarSensor, publishes a TelemetryEventMsg, and publishes
a THERMAL_OVER_LIMIT FaultEventMsg when the reading exceeds cfg.thermal_limit_c. It
acknowledges any CommandMsg targeting "thermal" with a command_ack telemetry event, and
emits periodic heartbeats. All decision logic is trivial; time is injected via Clock.

Satisfies: REQ-SAFE-HIGH-002 (thermal self-reporting), REQ-OPER-HIGH-002 (subsystem app).
"""

from __future__ import annotations

# stdlib
import threading
from dataclasses import dataclass

# internal
from flight.hal.interfaces import ScalarSensor
from flight.libs.bus import MessageBus, Subscription
from flight.libs.config import FaultConfig, PactConfig
from flight.libs.messages import CommandMsg, FaultEventMsg, HeartbeatMsg, TelemetryEventMsg
from flight.libs.time import Clock
from flight.libs.types import FaultCode, MessageType, Ok

SUBSYSTEM = "thermal"


@dataclass(frozen=True)
class ThermalApp:
    """Thermal housekeeping subsystem app (telemetry + over-limit fault + commandable)."""

    cfg: FaultConfig
    bus: MessageBus
    clock: Clock
    sensor: ScalarSensor
    commands: Subscription[CommandMsg]

    @staticmethod
    def from_config(
        cfg: PactConfig,
        bus: MessageBus,
        clock: Clock,
        sensor: ScalarSensor,
    ) -> ThermalApp:
        """Assemble a ThermalApp and subscribe it to inbound commands.

        Args:
            cfg: Top-level PactConfig (cfg.fault is retained for the limit + heartbeat).
            bus: The MessageBus to publish onto and subscribe to.
            clock: Injected Clock.
            sensor: The ScalarSensor reading temperature in Celsius.

        Returns:
            A ThermalApp holding a fresh CommandMsg subscription.
        """
        return ThermalApp(
            cfg=cfg.fault,
            bus=bus,
            clock=clock,
            sensor=sensor,
            commands=bus.subscribe(CommandMsg),
        )

    def sample(self) -> None:
        """Read the temperature, publish telemetry, and emit a fault if over the limit.

        On a sensor read error the cycle is skipped (no telemetry, no fault) -- a
        transient read failure surfaces as missing telemetry, which the watchdog/ground
        observe; there is no dedicated sensor-fault code.
        """
        result = self.sensor.read()
        if not isinstance(result, Ok):
            return
        temperature_c = result.value
        self.bus.publish(
            TelemetryEventMsg(
                msg_type=MessageType.TELEMETRY_EVENT,
                timestamp_utc=self.clock.wall_clock_iso(),
                subsystem=SUBSYSTEM,
                event_name="thermal_sample",
                payload={"temperature_c": temperature_c},
            )
        )
        if temperature_c > self.cfg.thermal_limit_c:
            self.bus.publish(
                FaultEventMsg(
                    msg_type=MessageType.FAULT_EVENT,
                    timestamp_utc=self.clock.wall_clock_iso(),
                    fault_code=FaultCode.THERMAL_OVER_LIMIT,
                    subsystem=SUBSYSTEM,
                    detail=(
                        f"temperature {temperature_c:.1f}C exceeds limit "
                        f"{self.cfg.thermal_limit_c:.1f}C"
                    ),
                )
            )

    def handle_commands(self) -> None:
        """Acknowledge each pending CommandMsg targeting this subsystem via telemetry."""
        while not self.commands.empty():
            command = self.commands.get_nowait()
            if command.target != SUBSYSTEM:
                continue
            self.bus.publish(
                TelemetryEventMsg(
                    msg_type=MessageType.TELEMETRY_EVENT,
                    timestamp_utc=self.clock.wall_clock_iso(),
                    subsystem=SUBSYSTEM,
                    event_name="command_ack",
                    payload={"command_id": command.command_id, "seq": command.seq},
                )
            )

    def run(self, stop_event: threading.Event) -> None:
        """Run the housekeeping loop until stop_event is set, with periodic heartbeats.

        Each iteration handles commands, samples, and emits a heartbeat every
        cfg.watchdog_interval_s; then waits one interval.

        Args:
            stop_event: threading.Event; the loop exits cleanly once it is set.
        """
        sequence = 0
        last_heartbeat = self.clock.monotonic_s()
        while not stop_event.is_set():
            self.handle_commands()
            self.sample()
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
```

- [ ] **Step 4: Export the app**

Overwrite `packages/flight/src/flight/thermal/__init__.py`:

```python
"""Thermal housekeeping subsystem: temperature telemetry + over-limit fault reporting."""

from flight.thermal.app import ThermalApp

__all__ = ["ThermalApp"]
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest packages/flight/tests/test_thermal_app.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add packages/flight/src/flight/thermal/app.py packages/flight/src/flight/thermal/__init__.py packages/flight/tests/test_thermal_app.py
git commit -m "feat(thermal): add minimal thermal housekeeping app"
```

---

### Task 3: ElectricalApp

**Files:**
- Create: `packages/flight/src/flight/electrical/app.py`
- Modify: `packages/flight/src/flight/electrical/__init__.py`
- Test: `packages/flight/tests/test_electrical_app.py`

- [ ] **Step 1: Write the failing test**

Create `packages/flight/tests/test_electrical_app.py`:

```python
"""Tests for the electrical housekeeping app (telemetry + threshold fault + command ack)."""

from flight.electrical.app import ElectricalApp
from flight.hal.drivers_sim import SimScalarSensor
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.messages import CommandMsg, FaultEventMsg, TelemetryEventMsg
from flight.libs.time import ManualClock
from flight.libs.types import FaultCode, MessageType


def _app(readings: list[float]) -> tuple[ElectricalApp, MessageBus]:
    """Assemble an ElectricalApp over a scripted power sensor and a fresh bus."""
    bus = MessageBus()
    sensor = SimScalarSensor(readings)
    app = ElectricalApp.from_config(PactConfig(), bus, ManualClock(), sensor)
    return app, bus


def test_nominal_reading_publishes_telemetry_no_fault() -> None:
    """A power draw below the limit publishes telemetry and no fault."""
    app, bus = _app([30.0])
    telem = bus.subscribe(TelemetryEventMsg)
    fault = bus.subscribe(FaultEventMsg)
    app.sample()
    assert not telem.empty()
    event = telem.get_nowait()
    assert event.subsystem == "electrical"
    assert event.payload["power_w"] == 30.0
    assert fault.empty()


def test_over_limit_reading_publishes_power_fault() -> None:
    """A power draw above power_limit_w (55.0) publishes POWER_OVER_LIMIT."""
    app, bus = _app([70.0])
    fault = bus.subscribe(FaultEventMsg)
    app.sample()
    assert not fault.empty()
    event = fault.get_nowait()
    assert event.fault_code is FaultCode.POWER_OVER_LIMIT
    assert event.subsystem == "electrical"


def test_command_targeting_electrical_is_acknowledged() -> None:
    """A CommandMsg targeting 'electrical' produces a command_ack telemetry event."""
    app, bus = _app([30.0])
    telem = bus.subscribe(TelemetryEventMsg)
    bus.publish(
        CommandMsg(
            msg_type=MessageType.COMMAND,
            timestamp_utc="t",
            target="electrical",
            command_id="ping",
            params={},
            source="ground",
            seq=1,
        )
    )
    app.handle_commands()
    assert not telem.empty()
    ack = telem.get_nowait()
    assert ack.event_name == "command_ack"
    assert ack.payload["command_id"] == "ping"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/flight/tests/test_electrical_app.py -v`
Expected: FAIL (no module `flight.electrical.app`).

- [ ] **Step 3: Write the implementation**

Create `packages/flight/src/flight/electrical/app.py`:

```python
"""Electrical housekeeping app: power telemetry, over-limit fault, command ack.

Minimal subsystem app proving the electrical node is in the topology: each cycle it
samples a power draw via a ScalarSensor, publishes a TelemetryEventMsg, and publishes a
POWER_OVER_LIMIT FaultEventMsg when the reading exceeds cfg.power_limit_w. It
acknowledges any CommandMsg targeting "electrical" with a command_ack telemetry event,
and emits periodic heartbeats. All decision logic is trivial; time is injected via Clock.

Satisfies: REQ-SAFE-HIGH-002 (power self-reporting), REQ-OPER-HIGH-002 (subsystem app).
"""

from __future__ import annotations

# stdlib
import threading
from dataclasses import dataclass

# internal
from flight.hal.interfaces import ScalarSensor
from flight.libs.bus import MessageBus, Subscription
from flight.libs.config import FaultConfig, PactConfig
from flight.libs.messages import CommandMsg, FaultEventMsg, HeartbeatMsg, TelemetryEventMsg
from flight.libs.time import Clock
from flight.libs.types import FaultCode, MessageType, Ok

SUBSYSTEM = "electrical"


@dataclass(frozen=True)
class ElectricalApp:
    """Electrical housekeeping subsystem app (telemetry + over-limit fault + commandable)."""

    cfg: FaultConfig
    bus: MessageBus
    clock: Clock
    sensor: ScalarSensor
    commands: Subscription[CommandMsg]

    @staticmethod
    def from_config(
        cfg: PactConfig,
        bus: MessageBus,
        clock: Clock,
        sensor: ScalarSensor,
    ) -> ElectricalApp:
        """Assemble an ElectricalApp and subscribe it to inbound commands.

        Args:
            cfg: Top-level PactConfig (cfg.fault is retained for the limit + heartbeat).
            bus: The MessageBus to publish onto and subscribe to.
            clock: Injected Clock.
            sensor: The ScalarSensor reading power draw in Watts.

        Returns:
            An ElectricalApp holding a fresh CommandMsg subscription.
        """
        return ElectricalApp(
            cfg=cfg.fault,
            bus=bus,
            clock=clock,
            sensor=sensor,
            commands=bus.subscribe(CommandMsg),
        )

    def sample(self) -> None:
        """Read the power draw, publish telemetry, and emit a fault if over the limit.

        On a sensor read error the cycle is skipped (no telemetry, no fault) -- a
        transient read failure surfaces as missing telemetry, which the watchdog/ground
        observe; there is no dedicated sensor-fault code.
        """
        result = self.sensor.read()
        if not isinstance(result, Ok):
            return
        power_w = result.value
        self.bus.publish(
            TelemetryEventMsg(
                msg_type=MessageType.TELEMETRY_EVENT,
                timestamp_utc=self.clock.wall_clock_iso(),
                subsystem=SUBSYSTEM,
                event_name="electrical_sample",
                payload={"power_w": power_w},
            )
        )
        if power_w > self.cfg.power_limit_w:
            self.bus.publish(
                FaultEventMsg(
                    msg_type=MessageType.FAULT_EVENT,
                    timestamp_utc=self.clock.wall_clock_iso(),
                    fault_code=FaultCode.POWER_OVER_LIMIT,
                    subsystem=SUBSYSTEM,
                    detail=(
                        f"power {power_w:.1f}W exceeds limit {self.cfg.power_limit_w:.1f}W"
                    ),
                )
            )

    def handle_commands(self) -> None:
        """Acknowledge each pending CommandMsg targeting this subsystem via telemetry."""
        while not self.commands.empty():
            command = self.commands.get_nowait()
            if command.target != SUBSYSTEM:
                continue
            self.bus.publish(
                TelemetryEventMsg(
                    msg_type=MessageType.TELEMETRY_EVENT,
                    timestamp_utc=self.clock.wall_clock_iso(),
                    subsystem=SUBSYSTEM,
                    event_name="command_ack",
                    payload={"command_id": command.command_id, "seq": command.seq},
                )
            )

    def run(self, stop_event: threading.Event) -> None:
        """Run the housekeeping loop until stop_event is set, with periodic heartbeats.

        Each iteration handles commands, samples, and emits a heartbeat every
        cfg.watchdog_interval_s; then waits one interval.

        Args:
            stop_event: threading.Event; the loop exits cleanly once it is set.
        """
        sequence = 0
        last_heartbeat = self.clock.monotonic_s()
        while not stop_event.is_set():
            self.handle_commands()
            self.sample()
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
```

- [ ] **Step 4: Export the app**

Overwrite `packages/flight/src/flight/electrical/__init__.py`:

```python
"""Electrical housekeeping subsystem: power telemetry + over-limit fault reporting."""

from flight.electrical.app import ElectricalApp

__all__ = ["ElectricalApp"]
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest packages/flight/tests/test_electrical_app.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add packages/flight/src/flight/electrical/app.py packages/flight/src/flight/electrical/__init__.py packages/flight/tests/test_electrical_app.py
git commit -m "feat(electrical): add minimal electrical housekeeping app"
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
- `ruff format --check packages` -> all files already formatted. If any new module/test would be reformatted, run `uv run ruff format packages` and commit with `style: ruff-format housekeeping apps`.
- `mypy packages` -> Success (now 105 source files: 101 + scalar interface + sim scalar + real scalar + thermal app + electrical app... count is approximate; all green is what matters).
- `lint-imports` -> Contracts: 7 kept, 0 broken. (Confirms thermal/electrical import only hal.interfaces + libs, do not import each other, and the scalar drivers stay independent.)
- `pytest packages -m "not e2e"` -> 166 passed, 1 skipped (155 + 11 new: 4 scalar + 4 thermal + 3 electrical).

- [ ] **Step 2: Commit any formatting fix (only if Step 1 required one)**

```bash
git add packages/flight/src/flight packages/flight/tests
git commit -m "style: ruff-format housekeeping apps"
```

---

## HARD RULES for the implementer

- Touch ONLY the files named in Tasks 1-3. Edits to the three HAL `__init__.py` are additive (add the scalar entries; preserve all existing entries/ordering).
- Do NOT modify `src/pact/**` (additive migration). Do NOT create or build `flight/mechanical` (out of scope this phase). Do NOT stage the pre-existing dirty working-tree entries (`src/pact/fault/detector.py`, `tests/**`, `.idea/*`, `.claude/settings.local.json`, `.coverage`, `bash.exe.stackdump`).
- Commits are LOCAL only; do not push.
- PowerShell/Windows: `uv run ...` for all gates; `git -m` single-quoted strings (no here-strings).
- Python 3.14 / PEP 758: never add parens to except clauses. Use `from __future__ import annotations` for `-> ThermalApp`/`-> ElectricalApp`.
- If mypy reports `no-any-return`, assign to a locally-annotated variable first — never add `# type: ignore`. If a gate fails, fix the cause; never weaken a test assertion.

## Self-Review (spec coverage)

- Thermal minimal app: heartbeat + temperature telemetry + THERMAL_OVER_LIMIT self-report + commandable ack. ✓ Task 2.
- Electrical minimal app: heartbeat + power telemetry + POWER_OVER_LIMIT self-report + commandable ack. ✓ Task 3.
- Shared ScalarSensor HAL + sim/real drivers. ✓ Task 1.
- Faults route to SAFE via the existing FDIR policy (no fault-subsystem change needed). ✓ (THERMAL_OVER_LIMIT/POWER_OVER_LIMIT already in SAFE_TRIGGERING_FAULTS).
- Layering preserved (peers don't cross-import; depend only on hal.interfaces + libs). ✓ Task 4 `lint-imports`.
- mechanical deliberately deferred. ✓ HARD RULES.
```
