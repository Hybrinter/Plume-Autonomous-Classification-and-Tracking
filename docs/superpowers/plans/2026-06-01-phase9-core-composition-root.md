# Phase 9: Core Composition Root + Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `flight/core` composition root — a driver-agnostic `build_apps()` that wires all five subsystem apps over one bus/clock, a thread-based `Scheduler` that runs them, and a flight `main()` entry that constructs real drivers and starts the system.

**Architecture:** `composition.py` defines a `Drivers` bundle and `build_apps(config, bus, clock, drivers, monitored) -> SystemApps`; it imports only HAL Protocols + the apps (never concrete drivers), so the same wiring serves both the flight entry and the SIL (Phase 10). `scheduler.py` defines a `RunnableApp` Protocol and a `Scheduler` that runs each app's `run(stop_event)` in a daemon thread (the bus is in-process, so threads share it). `main.py` is the flight composition root: it constructs the real drivers + the ONNX detector, calls `build_apps`, and runs the scheduler. The flight entry is runtime-only (real drivers/onnxruntime are not present in CI), so it is tested for importability and type-correctness; `build_apps` and `Scheduler` are fully tested with sim drivers and a fake runnable.

**Tech Stack:** Python 3.14, frozen dataclasses, `Protocol`-based wiring, threads, typed `MessageBus`, injected `Clock`. mypy --strict, ruff (line-length 100), import-linter, pytest.

---

## Context the implementer needs

**Verified app constructors (all exist, signatures confirmed):**
- `PayloadApp.from_config(cfg: PactConfig, sensor: ImagingSensor, gimbal: GimbalActuator, detector: DetectorBackend, bus: MessageBus, clock: Clock) -> PayloadApp` (`flight.payload.app`).
- `FaultApp.from_config(cfg: PactConfig, bus: MessageBus, clock: Clock, monitored: tuple[str, ...]) -> FaultApp` (`flight.fault.app`). Has `run(stop_event)`.
- `IssIfaceApp.from_config(cfg: PactConfig, bus: MessageBus, clock: Clock, link: StationLink) -> IssIfaceApp` (`flight.iss_iface.app`). Has `run(stop_event)`.
- `ThermalApp.from_config(cfg, bus, clock, sensor: ScalarSensor) -> ThermalApp` (`flight.thermal.app`). Has `run(stop_event)`.
- `ElectricalApp.from_config(cfg, bus, clock, sensor: ScalarSensor) -> ElectricalApp` (`flight.electrical.app`). Has `run(stop_event)`.
- `PayloadApp.run(stop_event: threading.Event) -> None` exists too. So all five apps share the `run(stop_event)` shape.

**HAL Protocols + detector + drivers:**
- `flight.hal.interfaces`: `ImagingSensor`, `GimbalActuator`, `ScalarSensor`, `StationLink`.
- `flight.payload.model`: `DetectorBackend` (Protocol), `OnnxDetector(model_path, ...)` (lazy onnxruntime), `ScriptedDetector(mask, ...)`.
- `flight.hal.drivers_real`: `RealSensor` (lazy PySpin -> ImportError without SDK), `RealGimbal`, `RealStationLink`, `RealScalarSensor` (these three are plain stubs, no SDK).
- `flight.hal.drivers_sim`: `SimSensor`, `SimGimbal`, `SimStationLink`, `SimScalarSensor` (used by TESTS here; the real composition is in Phase 10's SIL).

**Config / bus / clock:**
- `flight.core.config_loader.load_config(path, override=None) -> Result[PactConfig, str]`.
- `flight.libs.bus.MessageBus`; `flight.libs.time.RealClock`, `ManualClock`, `Clock`.
- `flight.libs.config.PactConfig`.

**Layering (import-linter), must hold:**
- `flight.core` is the top layer; it MAY import the subsystem apps, `flight.hal.interfaces`, `flight.libs.*`, and (in `main.py` only) `flight.hal.drivers_real`. `flight.core` is NOT in the `drivers-from-composition-roots-only` forbidden-source list, so importing `drivers_real` is allowed.
- `composition.py` and `scheduler.py` must NOT import any concrete driver (only `main.py` does).
- Tests may import `flight.hal.drivers_sim` and `ScriptedDetector` freely.

**mypy note (carried):** `uv run mypy packages` resolves cross-package `flight.*` to `Any`; if a method returning a generic-parameterized-by-imported-type trips `no-any-return`, assign to a locally-annotated variable first — never `# type: ignore`.

---

### Task 1: Scheduler (RunnableApp protocol + thread runner)

**Files:**
- Create: `packages/flight/src/flight/core/scheduler.py`
- Test: `packages/flight/tests/test_scheduler.py`

- [ ] **Step 1: Write the failing test**

Create `packages/flight/tests/test_scheduler.py`:

```python
"""Tests for the thread-based subsystem scheduler."""

import threading

from flight.core.scheduler import RunnableApp, Scheduler


class _BlockingApp:
    """RunnableApp that signals it started, then blocks until stopped."""

    def __init__(self, started: threading.Event) -> None:
        self._started = started

    def run(self, stop_event: threading.Event) -> None:
        """Signal startup, then wait until the scheduler sets stop_event."""
        self._started.set()
        stop_event.wait()


def test_blocking_app_satisfies_runnable_protocol() -> None:
    """_BlockingApp conforms to RunnableApp at runtime."""
    app: RunnableApp = _BlockingApp(threading.Event())
    assert isinstance(app, RunnableApp)


def test_scheduler_starts_apps_in_threads() -> None:
    """start() launches each app's run() in a thread that actually executes."""
    started = threading.Event()
    scheduler = Scheduler([("worker", _BlockingApp(started))])
    scheduler.start()
    try:
        assert started.wait(timeout=2.0)  # the worker thread ran
        assert scheduler.is_running()
    finally:
        scheduler.stop(timeout=2.0)


def test_scheduler_stop_joins_threads() -> None:
    """stop() sets the shared stop event and joins all threads."""
    started = threading.Event()
    scheduler = Scheduler([("worker", _BlockingApp(started))])
    scheduler.start()
    assert started.wait(timeout=2.0)
    scheduler.stop(timeout=2.0)
    assert not scheduler.is_running()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/flight/tests/test_scheduler.py -v`
Expected: FAIL (no module `flight.core.scheduler`).

- [ ] **Step 3: Write the implementation**

Create `packages/flight/src/flight/core/scheduler.py`:

```python
"""Thread-based scheduler for subsystem apps.

The message bus is in-process (queue.Queue transport), so subsystem apps run as
daemon threads that share it. The scheduler owns one stop Event, launches each app's
run(stop_event) in a named thread, and joins them on stop. Each app owns its own loop
and internal state; the scheduler only starts and stops them.

Contains:
  - RunnableApp: the Protocol every schedulable app satisfies (run(stop_event)).
  - Scheduler: start() launches threads, stop() signals + joins, is_running() reports liveness.
"""

from __future__ import annotations

# stdlib
import threading
from typing import Protocol, runtime_checkable


@runtime_checkable
class RunnableApp(Protocol):
    """A subsystem app the scheduler can run: a single blocking run(stop_event) loop."""

    def run(self, stop_event: threading.Event) -> None:
        """Run until stop_event is set."""
        ...


class Scheduler:
    """Runs each registered app's run(stop_event) in its own daemon thread."""

    def __init__(self, apps: list[tuple[str, RunnableApp]]) -> None:
        """Register the apps to schedule.

        Args:
            apps: (name, app) pairs; name labels the thread for diagnostics.
        """
        self._apps = apps
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        """Launch each app's run(stop_event) in a named daemon thread."""
        for name, app in self._apps:
            thread = threading.Thread(
                target=app.run, args=(self._stop,), name=name, daemon=True
            )
            thread.start()
            self._threads.append(thread)

    def stop(self, timeout: float = 5.0) -> None:
        """Signal every app to stop and join each thread up to timeout seconds.

        The joined (now-dead) threads are retained so is_running() reports liveness
        honestly after stop rather than reading an emptied list.
        """
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=timeout)

    def is_running(self) -> bool:
        """Return True if any scheduled thread is still alive."""
        return any(thread.is_alive() for thread in self._threads)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest packages/flight/tests/test_scheduler.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/flight/src/flight/core/scheduler.py packages/flight/tests/test_scheduler.py
git commit -m "feat(core): add thread-based subsystem scheduler"
```

---

### Task 2: Composition (Drivers bundle + build_apps)

**Files:**
- Create: `packages/flight/src/flight/core/composition.py`
- Test: `packages/flight/tests/test_composition.py`

- [ ] **Step 1: Write the failing test**

Create `packages/flight/tests/test_composition.py`:

```python
"""Tests for the driver-agnostic composition root (build_apps)."""

import numpy as np
from flight.core.composition import MONITORED_SUBSYSTEMS, Drivers, SystemApps, build_apps
from flight.electrical.app import ElectricalApp
from flight.fault.app import FaultApp
from flight.hal.drivers_sim import SimGimbal, SimScalarSensor, SimSensor, SimStationLink
from flight.iss_iface.app import IssIfaceApp
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.time import ManualClock
from flight.payload.app import PayloadApp
from flight.payload.model import ScriptedDetector
from flight.thermal.app import ThermalApp


def _drivers() -> Drivers:
    """Bundle sim drivers + a scripted detector for composition testing."""
    return Drivers(
        sensor=SimSensor([]),
        gimbal=SimGimbal(),
        detector=ScriptedDetector(np.zeros((256, 256), dtype=np.float32)),
        station=SimStationLink([]),
        thermal_sensor=SimScalarSensor([20.0]),
        power_sensor=SimScalarSensor([10.0]),
    )


def test_build_apps_wires_all_five_subsystems() -> None:
    """build_apps constructs all five subsystem apps over the shared bus/clock."""
    apps = build_apps(PactConfig(), MessageBus(), ManualClock(), _drivers(), MONITORED_SUBSYSTEMS)
    assert isinstance(apps, SystemApps)
    assert isinstance(apps.payload, PayloadApp)
    assert isinstance(apps.fault, FaultApp)
    assert isinstance(apps.iss_iface, IssIfaceApp)
    assert isinstance(apps.thermal, ThermalApp)
    assert isinstance(apps.electrical, ElectricalApp)


def test_monitored_subsystems_are_the_heartbeat_producers() -> None:
    """The default monitored set is exactly the four heartbeat-emitting subsystems."""
    assert set(MONITORED_SUBSYSTEMS) == {"payload", "iss_iface", "thermal", "electrical"}


def test_build_apps_shares_one_bus() -> None:
    """All apps are wired to the same bus instance passed in."""
    bus = MessageBus()
    apps = build_apps(PactConfig(), bus, ManualClock(), _drivers(), MONITORED_SUBSYSTEMS)
    assert apps.payload.bus is bus
    assert apps.fault.bus is bus
    assert apps.thermal.bus is bus
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/flight/tests/test_composition.py -v`
Expected: FAIL (no module `flight.core.composition`).

- [ ] **Step 3: Write the implementation**

Create `packages/flight/src/flight/core/composition.py`:

```python
"""Driver-agnostic composition root: wires the subsystem apps over one bus + clock.

build_apps() is the single place the full app topology is assembled. It depends only
on HAL Protocols and the apps -- never on concrete drivers -- so the same wiring serves
the flight entry (real drivers, in core/main.py) and the SIL (sim drivers, in
sim/sil). The caller constructs the Drivers bundle and owns the bus and clock.

Contains:
  - Drivers: the bundle of injected HAL drivers + the detector backend.
  - SystemApps: the five constructed subsystem apps.
  - MONITORED_SUBSYSTEMS: the heartbeat-emitting subsystems the FDIR watchdog watches.
  - build_apps: construct every app from config + bus + clock + drivers.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass

# internal
from flight.electrical.app import ElectricalApp
from flight.fault.app import FaultApp
from flight.hal.interfaces import GimbalActuator, ImagingSensor, ScalarSensor, StationLink
from flight.iss_iface.app import IssIfaceApp
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.time import Clock
from flight.payload.app import PayloadApp
from flight.payload.model import DetectorBackend
from flight.thermal.app import ThermalApp

# The subsystems that run persistent loops and emit heartbeats; the FDIR watchdog
# monitors exactly these (the fault subsystem does not monitor itself).
MONITORED_SUBSYSTEMS: tuple[str, ...] = ("payload", "iss_iface", "thermal", "electrical")


@dataclass(frozen=True)
class Drivers:
    """Bundle of injected HAL drivers + the detector backend for one composition.

    The composition root (flight entry or SIL) constructs the concrete implementations;
    build_apps consumes only the Protocol types.
    """

    sensor: ImagingSensor
    gimbal: GimbalActuator
    detector: DetectorBackend
    station: StationLink
    thermal_sensor: ScalarSensor
    power_sensor: ScalarSensor


@dataclass(frozen=True)
class SystemApps:
    """The five constructed subsystem apps, sharing one bus and clock."""

    payload: PayloadApp
    fault: FaultApp
    iss_iface: IssIfaceApp
    thermal: ThermalApp
    electrical: ElectricalApp


def build_apps(
    config: PactConfig,
    bus: MessageBus,
    clock: Clock,
    drivers: Drivers,
    monitored: tuple[str, ...],
) -> SystemApps:
    """Construct every subsystem app wired to the shared bus and clock.

    Args:
        config: The validated PactConfig.
        bus: The single MessageBus all apps publish to / subscribe from.
        clock: The injected Clock (RealClock in flight, ManualClock in SIL/tests).
        drivers: The HAL driver bundle (real or sim) plus the detector backend.
        monitored: Subsystem names the FDIR watchdog should watch (use MONITORED_SUBSYSTEMS).

    Returns:
        A SystemApps with all five apps constructed.
    """
    return SystemApps(
        payload=PayloadApp.from_config(
            config, drivers.sensor, drivers.gimbal, drivers.detector, bus, clock
        ),
        fault=FaultApp.from_config(config, bus, clock, monitored),
        iss_iface=IssIfaceApp.from_config(config, bus, clock, drivers.station),
        thermal=ThermalApp.from_config(config, bus, clock, drivers.thermal_sensor),
        electrical=ElectricalApp.from_config(config, bus, clock, drivers.power_sensor),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest packages/flight/tests/test_composition.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/flight/src/flight/core/composition.py packages/flight/tests/test_composition.py
git commit -m "feat(core): add driver-agnostic composition root (build_apps)"
```

---

### Task 3: Flight entry (main.py) + core exports

**Files:**
- Create: `packages/flight/src/flight/core/main.py`
- Modify: `packages/flight/src/flight/core/__init__.py`
- Test: `packages/flight/tests/test_core_main.py`

- [ ] **Step 1: Write the failing test**

Create `packages/flight/tests/test_core_main.py`:

```python
"""Smoke tests for the flight entry module (importable + exposes its API).

build_flight_system() and main() construct real drivers (RealSensor lazy-imports
PySpin; OnnxDetector lazy-imports onnxruntime), so they are NOT executed here -- CI has
neither SDK. These tests assert the module imports and exposes the expected callables;
the wiring itself is exercised by the SIL integration (Phase 10) via build_apps.
"""

from flight.core import build_flight_system, main
from flight.core.scheduler import Scheduler


def test_core_exposes_entry_callables() -> None:
    """flight.core re-exports the flight entry points."""
    assert callable(build_flight_system)
    assert callable(main)


def test_scheduler_is_importable_from_core() -> None:
    """The scheduler type used by main is importable (sanity for the entry wiring)."""
    assert Scheduler is not None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/flight/tests/test_core_main.py -v`
Expected: FAIL (ImportError: cannot import `build_flight_system`/`main` from `flight.core`).

- [ ] **Step 3: Write the implementation**

Create `packages/flight/src/flight/core/main.py`:

```python
"""Flight composition root: construct real drivers and run the subsystem scheduler.

This is the production entry on the payload computer. It loads config, constructs the
real HAL drivers and the ONNX detector, wires every app via build_apps, and runs them
under the thread Scheduler until interrupted. Real drivers and onnxruntime are present
only on flight hardware, so this module is constructed/run at runtime, not in CI; the
driver-agnostic wiring it relies on (build_apps) is unit-tested with sim drivers.

Contains:
  - build_flight_system: construct the real Drivers bundle and the SystemApps.
  - main: load config, build the system, and run the scheduler until interrupted.
"""

from __future__ import annotations

# stdlib
import threading

# internal
from flight.core.composition import MONITORED_SUBSYSTEMS, Drivers, SystemApps, build_apps
from flight.core.config_loader import load_config
from flight.core.scheduler import Scheduler
from flight.hal.drivers_real import RealGimbal, RealScalarSensor, RealSensor, RealStationLink
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.time import Clock, RealClock
from flight.libs.types import Ok
from flight.payload.model import OnnxDetector


def build_flight_system(config: PactConfig, bus: MessageBus, clock: Clock) -> SystemApps:
    """Construct the real-driver Drivers bundle and wire the SystemApps.

    Args:
        config: The validated PactConfig.
        bus: The shared MessageBus.
        clock: The injected Clock (RealClock in production).

    Returns:
        The wired SystemApps.

    Notes:
        RealSensor lazily imports PySpin and OnnxDetector lazily imports onnxruntime;
        both raise ImportError if the SDK is absent. This function therefore runs only
        on flight hardware.
    """
    drivers = Drivers(
        sensor=RealSensor(),
        gimbal=RealGimbal(),
        detector=OnnxDetector(config.inference.model_path),
        station=RealStationLink(),
        thermal_sensor=RealScalarSensor(),
        power_sensor=RealScalarSensor(),
    )
    return build_apps(config, bus, clock, drivers, MONITORED_SUBSYSTEMS)


def main(config_path: str = "config/default.toml") -> None:
    """Load config, build the flight system, and run the scheduler until interrupted.

    Args:
        config_path: Path to the TOML config file.

    Raises:
        SystemExit: If config loading fails (unrecoverable startup error).
    """
    result = load_config(config_path)
    if not isinstance(result, Ok):
        raise SystemExit(f"config load failed: {result.error}")
    config = result.value

    bus = MessageBus()
    clock: Clock = RealClock()
    apps = build_flight_system(config, bus, clock)

    scheduler = Scheduler(
        [
            ("payload", apps.payload),
            ("fault", apps.fault),
            ("iss_iface", apps.iss_iface),
            ("thermal", apps.thermal),
            ("electrical", apps.electrical),
        ]
    )
    scheduler.start()
    try:
        threading.Event().wait()  # run until the process is signaled/interrupted
    except KeyboardInterrupt:
        scheduler.stop()
```

- [ ] **Step 4: Update core exports**

Overwrite `packages/flight/src/flight/core/__init__.py`:

```python
"""Compute / C&DH host: config loading, the composition root, and scheduling."""

from flight.core.composition import MONITORED_SUBSYSTEMS, Drivers, SystemApps, build_apps
from flight.core.config_loader import load_config
from flight.core.main import build_flight_system, main
from flight.core.scheduler import RunnableApp, Scheduler

__all__ = [
    "MONITORED_SUBSYSTEMS",
    "Drivers",
    "RunnableApp",
    "Scheduler",
    "SystemApps",
    "build_apps",
    "build_flight_system",
    "load_config",
    "main",
]
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest packages/flight/tests/test_core_main.py -v`
Expected: 2 passed. (Importing `flight.core.main` does NOT trigger PySpin/onnxruntime — those load only when `build_flight_system`/`main` is called.)

- [ ] **Step 6: Commit**

```bash
git add packages/flight/src/flight/core/main.py packages/flight/src/flight/core/__init__.py packages/flight/tests/test_core_main.py
git commit -m "feat(core): add flight entry (build_flight_system + main) and exports"
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
- `ruff format --check packages` -> all files already formatted (else `uv run ruff format packages` + a `style:` commit).
- `mypy packages` -> Success (all green; ~112 source files).
- `lint-imports` -> Contracts: 7 kept, 0 broken. (Critical: `flight.core.composition`/`scheduler` import NO concrete driver; only `flight.core.main` imports `flight.hal.drivers_real`, which is allowed for core.)
- `pytest packages -m "not e2e"` -> 174 passed, 1 skipped (166 + 8 new: 3 scheduler + 3 composition + 2 core_main).

- [ ] **Step 2: Commit any formatting fix (only if Step 1 required one)**

```bash
git add packages/flight/src/flight/core packages/flight/tests
git commit -m "style: ruff-format core composition root"
```

---

## HARD RULES for the implementer

- Touch ONLY the files named in Tasks 1-3.
- Do NOT modify `src/pact/**` (additive migration). Do NOT stage the pre-existing dirty working-tree entries (`src/pact/fault/detector.py`, `tests/**`, `.idea/*`, `.claude/settings.local.json`, `.coverage`, `bash.exe.stackdump`).
- Commits are LOCAL only; do not push.
- `composition.py` and `scheduler.py` must NOT import any concrete driver (`drivers_real`/`drivers_sim`). Only `main.py` imports `drivers_real`. If `lint-imports` breaks, you violated this.
- Do NOT call `build_flight_system()` or `main()` in any test (they construct PySpin/onnxruntime-backed drivers and will ImportError in CI). Test only importability + `build_apps`/`Scheduler` with sim/fake objects.
- PowerShell/Windows: `uv run ...` for all gates; `git -m` single-quoted strings (no here-strings).
- Python 3.14 / PEP 758: never add parens to except clauses. Use `from __future__ import annotations`.
- If mypy reports `no-any-return`, assign to a locally-annotated variable first — never `# type: ignore`. If a gate fails, fix the cause; never weaken a test assertion.

## Self-Review (spec coverage)

- Driver-agnostic wiring of all five apps over one bus/clock. ✓ Task 2 (`build_apps`).
- Thread scheduler running each app's `run(stop_event)`. ✓ Task 1 (`Scheduler`).
- Flight composition root constructing real drivers. ✓ Task 3 (`main.py`).
- Layering: composition/scheduler driver-free; only main imports real drivers. ✓ Task 4 `lint-imports`.
- Reusable by Phase 10 SIL (sim drivers + `build_apps`). ✓ (`build_apps` is driver-agnostic).
```
