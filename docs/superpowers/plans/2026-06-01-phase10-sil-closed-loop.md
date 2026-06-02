# Phase 10: SIL Closed-Loop Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the software-in-the-loop (SIL) harness in the `sim` package that runs the real flight apps against sim drivers, and prove the closed loop end-to-end: a plume scene drives payload detection -> gimbal command + telemetry, and a thermal over-limit drives the FDIR app to publish SAFE.

**Architecture:** `sim/scene` generates a synthetic plume scene (zeroed raw frames + a `ScriptedDetector` whose fixed mask yields one stable central blob). `sim/sil` constructs the sim drivers, calls the Phase-9 driver-agnostic `build_apps()`, and exposes a deterministic `SilHarness` that steps every app once per cycle over the shared in-process bus (no threads — fully deterministic). The harness stands in for each app's `run()` heartbeat so the FDIR watchdog reflects a live system. Two integration tests prove the data path (plume -> command + telemetry) and the FDIR path (thermal fault -> SAFE). The digital twin (`sim/twin`) stays an empty scaffold; `SimGimbal`'s delta integration is sufficient pointing dynamics for this SIL.

**Tech Stack:** Python 3.14, numpy, frozen dataclasses, the typed `MessageBus`, `ManualClock`. mypy --strict, ruff (line-length 100), import-linter, pytest. The `sim` package depends on `pact-flight` (workspace), so it imports the flight apps, `flight.core.build_apps`, and `flight.hal.drivers_sim` directly.

---

## Context the implementer needs

**Reused Phase-9 surface (verified):**
- `flight.core.composition` — `Drivers(sensor, gimbal, detector, station, thermal_sensor, power_sensor)`, `SystemApps(payload, fault, iss_iface, thermal, electrical)`, `MONITORED_SUBSYSTEMS = ("payload", "iss_iface", "thermal", "electrical")`, `build_apps(config, bus, clock, drivers, monitored) -> SystemApps`.
- The apps' granular (deterministic) methods, used by the harness instead of `run()`:
  - `PayloadApp.controller.initial_state() -> ControlState`; `PayloadApp.process_frame(raw, state, now) -> tuple[ControlState, TickOutcome]`.
  - `FaultApp.initial_entries() -> dict[str, WatchdogEntry]`; `FaultApp.tick(entries, now) -> dict[str, WatchdogEntry]`.
  - `IssIfaceApp.tick() -> None`.
  - `ThermalApp.handle_commands() -> None`, `ThermalApp.sample() -> None`; same on `ElectricalApp`.

**Sim drivers + detector + messages:**
- `flight.hal.drivers_sim` — `SimSensor(frames: list[RawFrameMsg])` (replays then `Err(CAMERA_STALL)`), `SimGimbal()` (integrates deltas; `read_position() -> Result[GimbalPosition, FaultCode]`), `SimStationLink(inbound: list[CommandMsg])`, `SimScalarSensor(readings: list[float])` (replays, holds last).
- `flight.payload.model.ScriptedDetector(prob_mask, confidence_gate, min_blob_area_px)`.
- `flight.libs.messages` — `RawFrameMsg`, `ProcessedFrameMsg`, `InferenceResultMsg`, `GimbalCommandMsg`, `TelemetryEventMsg`, `FaultEventMsg`, `HeartbeatMsg`, `ModeChangeMsg`, `CommandMsg`.
- `flight.libs.types` — `Ok`, `MessageType`, `SystemMode`.
- `flight.libs.bus.MessageBus`; `flight.libs.time.ManualClock`; `flight.libs.config.PactConfig`.

**Why the harness publishes heartbeats:** in real operation each app emits its own `HeartbeatMsg` inside `run()`. The deterministic harness calls the granular methods (not `run()`), so it publishes one heartbeat per monitored subsystem per step to stand in for that liveness; otherwise the watchdog would trip after `watchdog_max_miss_count` steps and inject a spurious SAFE.

**Why this is deterministic (not threaded):** stepping every app once per cycle over the in-process bus, with `now` advanced explicitly, removes all timing nondeterminism. The threaded `Scheduler` (Phase 9) is for real operation; the SIL test uses single-threaded stepping.

**Layering / packaging:**
- `sim` may import `flight.*` freely (only `flight -> sim` is forbidden). `sim` must not import `tools`. `sim/sil` importing `flight.hal.drivers_sim` is allowed (the `drivers-from-composition-roots-only` contract constrains only `flight.*` modules, not `sim`).
- The integration test lives in `packages/sim/tests/` and is NOT marked `e2e` (it is in-process and fast), so it runs in the CI `pytest packages -m "not e2e"` gate.

**mypy note (carried):** `uv run mypy packages` resolves cross-package `flight.*` imports to `Any`; if a function returning a generic-parameterized-by-imported-type trips `no-any-return`, assign to a locally-annotated variable first — never `# type: ignore`.

---

### Task 1: Plume scene generation (sim/scene)

**Files:**
- Create: `packages/sim/src/sim/scene/plume.py`
- Modify: `packages/sim/src/sim/scene/__init__.py`
- Test: `packages/sim/tests/test_scene.py`

- [ ] **Step 1: Write the failing test**

Create `packages/sim/tests/test_scene.py`:

```python
"""Tests for SIL plume scene generation."""

import numpy as np
from flight.libs.messages import ProcessedFrameMsg
from flight.libs.types import MessageType, Ok
from sim.scene import build_frames, plume_detector


def test_build_frames_count_and_shape() -> None:
    """build_frames returns N frames each shaped (4, 256, 256)."""
    frames = build_frames(3)
    assert len(frames) == 3
    assert np.asarray(frames[0].raw_bands).shape == (4, 256, 256)
    assert frames[0].frame_id == 1
    assert frames[2].frame_id == 3


def test_plume_detector_finds_one_blob() -> None:
    """The scripted plume detector yields exactly one blob on a processed frame."""
    detector = plume_detector()
    frame = ProcessedFrameMsg(
        msg_type=MessageType.PROCESSED_FRAME,
        timestamp_utc="t",
        frame_id=1,
        tensor=np.zeros((4, 256, 256), dtype=np.float32),
        quality_flags=frozenset(),
        crop_origin_px=(0, 0),
        scale_factor=1.0,
    )
    result = detector.detect(frame)
    assert isinstance(result, Ok)
    assert len(result.value.blobs) == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/sim/tests/test_scene.py -v`
Expected: FAIL (no module `sim.scene.plume` / cannot import `build_frames`).

- [ ] **Step 3: Write the implementation**

Create `packages/sim/src/sim/scene/plume.py`:

```python
"""Plume scene generation for SIL: synthetic raw frames + a scripted plume detector.

The frames are zeroed (4, 256, 256) band stacks matching the payload's identity
calibration shape; the ScriptedDetector ignores the tensor content and detects from a
fixed probability mask, so a zeroed scene plus a plume mask yields a stable, strong
central blob every frame -- exactly what drives the gimbal arbiter to TRACKING.

Contains:
  - build_frames: N zeroed RawFrameMsg frames with monotonic frame_ids.
  - plume_detector: a ScriptedDetector whose mask yields one persistent central blob.
"""

from __future__ import annotations

# third-party
import numpy as np

# internal
from flight.libs.messages import RawFrameMsg
from flight.libs.types import MessageType
from flight.payload.model import ScriptedDetector

FRAME_BANDS = 4
FRAME_SIZE = 256


def build_frames(num_frames: int) -> list[RawFrameMsg]:
    """Build a list of zeroed raw frames for the SIL sensor to replay.

    Args:
        num_frames: Number of frames to generate.

    Returns:
        A list of num_frames RawFrameMsg, each a zeroed (4, 256, 256) float32 band
        stack with frame_id running 1..num_frames.
    """
    frames: list[RawFrameMsg] = []
    for frame_id in range(1, num_frames + 1):
        raw_bands = np.zeros(
            (FRAME_BANDS, FRAME_SIZE, FRAME_SIZE), dtype=np.float32
        )  # np.ndarray[float32, (C, H, W)]
        frames.append(
            RawFrameMsg(
                msg_type=MessageType.RAW_FRAME,
                timestamp_utc="2026-06-01T00:00:00.000Z",
                frame_id=frame_id,
                raw_bands=raw_bands,
                exposure_us=1000.0,
                gain_db=0.0,
                gimbal_az_deg=0.0,
                gimbal_el_deg=0.0,
            )
        )
    return frames


def plume_detector() -> ScriptedDetector:
    """Build a ScriptedDetector whose fixed mask yields one strong, stable central blob.

    Returns:
        A ScriptedDetector with a 50x50 unit-probability square (area 2500 px,
        confidence 1.0) centered in a 256x256 mask -- above the default gates.
    """
    mask = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.float32)  # np.ndarray[float32, (H, W)]
    mask[100:150, 100:150] = 1.0
    return ScriptedDetector(mask, confidence_gate=0.55, min_blob_area_px=15)
```

- [ ] **Step 4: Update the scene package exports**

Overwrite `packages/sim/src/sim/scene/__init__.py`:

```python
"""SIL scene generation: synthetic frames and scripted detections."""

from sim.scene.plume import build_frames, plume_detector

__all__ = ["build_frames", "plume_detector"]
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest packages/sim/tests/test_scene.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add packages/sim/src/sim/scene/plume.py packages/sim/src/sim/scene/__init__.py packages/sim/tests/test_scene.py
git commit -m "feat(sim): add plume scene generation for SIL"
```

---

### Task 2: SIL harness + closed-loop integration test (sim/sil)

**Files:**
- Create: `packages/sim/src/sim/sil/runner.py`
- Modify: `packages/sim/src/sim/sil/__init__.py`
- Test: `packages/sim/tests/test_sil_closed_loop.py`

- [ ] **Step 1: Write the failing test**

Create `packages/sim/tests/test_sil_closed_loop.py`:

```python
"""SIL closed-loop integration: the real flight apps over sim drivers via build_apps."""

from flight.libs.config import PactConfig
from flight.libs.messages import (
    FaultEventMsg,
    GimbalCommandMsg,
    InferenceResultMsg,
    ModeChangeMsg,
    TelemetryEventMsg,
)
from flight.libs.time import ManualClock
from flight.libs.types import Ok, SystemMode
from sim.scene import build_frames, plume_detector
from sim.sil import SilHarness, build_sil_system


def test_sil_nominal_closed_loop_tracks_plume() -> None:
    """A plume scene drives payload detection -> gimbal command + telemetry, no SAFE."""
    system = build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(8),
        plume_detector(),
        inbound_commands=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )
    cmd_sub = system.bus.subscribe(GimbalCommandMsg)
    inf_sub = system.bus.subscribe(InferenceResultMsg)
    telem_sub = system.bus.subscribe(TelemetryEventMsg)
    mode_sub = system.bus.subscribe(ModeChangeMsg)

    SilHarness(system).run_steps(8, dt=1.0)

    # Payload tracked the plume and commanded the gimbal off the origin.
    assert not cmd_sub.empty()
    position = system.gimbal.read_position()
    assert isinstance(position, Ok)
    assert (position.value.az_deg, position.value.el_deg) != (0.0, 0.0)

    # Inference ran once per frame.
    inference_count = 0
    while not inf_sub.empty():
        inf_sub.get_nowait()
        inference_count += 1
    assert inference_count == 8

    # Housekeeping telemetry flowed and the system stayed nominal (no SAFE).
    assert not telem_sub.empty()
    assert mode_sub.empty()


def test_sil_thermal_fault_drives_safe_mode() -> None:
    """A thermal over-limit self-reports a fault that the FDIR app routes to SAFE."""
    system = build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(6),
        plume_detector(),
        inbound_commands=[],
        thermal_readings=[25.0, 25.0, 95.0, 95.0, 95.0, 95.0],  # spikes over the 80C limit
        power_readings=[30.0],
    )
    fault_sub = system.bus.subscribe(FaultEventMsg)
    mode_sub = system.bus.subscribe(ModeChangeMsg)

    SilHarness(system).run_steps(6, dt=1.0)

    # Thermal self-reported the over-limit fault and FDIR commanded SAFE.
    assert not fault_sub.empty()
    assert not mode_sub.empty()
    assert mode_sub.get_nowait().new_mode is SystemMode.SAFE
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/sim/tests/test_sil_closed_loop.py -v`
Expected: FAIL (no module `sim.sil.runner` / cannot import `SilHarness`).

- [ ] **Step 3: Write the implementation**

Create `packages/sim/src/sim/sil/runner.py`:

```python
"""SIL runner: build the flight apps over sim drivers and step them deterministically.

build_sil_system constructs the sim drivers + scripted detector, bundles them as
Drivers, and calls the Phase-9 driver-agnostic build_apps -- so the SIL exercises the
exact same wiring the flight entry uses. SilHarness drives the apps single-threaded:
each step acquires + processes one frame, samples housekeeping, pumps the ISS bridge,
publishes per-subsystem liveness heartbeats, then runs the FDIR tick -- all over the
shared in-process bus, with `now` advanced explicitly for full determinism.

Contains:
  - SilSystem: the wired apps + bus + clock + the concrete sim drivers (for inspection).
  - build_sil_system: construct the sim drivers and wire the apps via build_apps.
  - SilHarness: deterministic single-threaded stepper (step / run_steps).
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass

# internal
from flight.core.composition import MONITORED_SUBSYSTEMS, Drivers, SystemApps, build_apps
from flight.fault.watchdog import WatchdogEntry
from flight.hal.drivers_sim import SimGimbal, SimScalarSensor, SimSensor, SimStationLink
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.messages import CommandMsg, HeartbeatMsg, RawFrameMsg
from flight.libs.time import ManualClock
from flight.libs.types import MessageType, Ok
from flight.payload.control import ControlState
from flight.payload.model import ScriptedDetector


@dataclass(frozen=True)
class SilSystem:
    """The wired SIL system: apps + shared bus/clock + the concrete sim drivers."""

    apps: SystemApps
    bus: MessageBus
    clock: ManualClock
    sensor: SimSensor
    gimbal: SimGimbal
    station: SimStationLink
    thermal_sensor: SimScalarSensor
    power_sensor: SimScalarSensor


def build_sil_system(
    config: PactConfig,
    clock: ManualClock,
    frames: list[RawFrameMsg],
    detector: ScriptedDetector,
    inbound_commands: list[CommandMsg],
    thermal_readings: list[float],
    power_readings: list[float],
) -> SilSystem:
    """Construct the sim drivers and wire the flight apps over a fresh bus via build_apps.

    Args:
        config: The PactConfig to wire the apps with.
        clock: The ManualClock shared by all apps (timestamps; the harness advances `now`).
        frames: Raw frames the SimSensor replays.
        detector: The ScriptedDetector backing the payload.
        inbound_commands: Commands the SimStationLink delivers via the ISS bridge.
        thermal_readings: Temperature readings the thermal sensor replays (Celsius).
        power_readings: Power readings the electrical sensor replays (Watts).

    Returns:
        A SilSystem holding the wired apps, the shared bus/clock, and the sim drivers.
    """
    bus = MessageBus()
    sensor = SimSensor(frames)
    gimbal = SimGimbal()
    station = SimStationLink(inbound_commands)
    thermal_sensor = SimScalarSensor(thermal_readings)
    power_sensor = SimScalarSensor(power_readings)
    drivers = Drivers(
        sensor=sensor,
        gimbal=gimbal,
        detector=detector,
        station=station,
        thermal_sensor=thermal_sensor,
        power_sensor=power_sensor,
    )
    apps = build_apps(config, bus, clock, drivers, MONITORED_SUBSYSTEMS)
    return SilSystem(
        apps=apps,
        bus=bus,
        clock=clock,
        sensor=sensor,
        gimbal=gimbal,
        station=station,
        thermal_sensor=thermal_sensor,
        power_sensor=power_sensor,
    )


class SilHarness:
    """Deterministic single-threaded driver for a SilSystem (no scheduler threads)."""

    def __init__(self, system: SilSystem) -> None:
        """Seed the payload control state and the FDIR watchdog entries.

        Args:
            system: The wired SilSystem to drive.
        """
        self._system = system
        self._payload_state: ControlState = system.apps.payload.controller.initial_state()
        self._fault_entries: dict[str, WatchdogEntry] = system.apps.fault.initial_entries()

    def step(self, now: float) -> None:
        """Advance every subsystem one cycle over the shared bus.

        Order: acquire + process one payload frame (if available) -> housekeeping
        handle-commands + sample -> ISS bridge pump -> publish per-subsystem liveness
        heartbeats -> FDIR tick (drains heartbeats + faults, publishes any SAFE).

        Args:
            now: Monotonic seconds for the arbiter and watchdog (advanced by the caller).
        """
        system = self._system
        apps = system.apps

        acquired = system.sensor.acquire_frame()
        if isinstance(acquired, Ok):
            self._payload_state, _ = apps.payload.process_frame(
                acquired.value, self._payload_state, now
            )

        apps.thermal.handle_commands()
        apps.thermal.sample()
        apps.electrical.handle_commands()
        apps.electrical.sample()

        apps.iss_iface.tick()

        for subsystem in MONITORED_SUBSYSTEMS:
            system.bus.publish(
                HeartbeatMsg(
                    msg_type=MessageType.HEARTBEAT,
                    timestamp_utc=system.clock.wall_clock_iso(),
                    subsystem=subsystem,
                    sequence=0,
                )
            )

        self._fault_entries = apps.fault.tick(self._fault_entries, now)

    def run_steps(self, count: int, dt: float = 1.0) -> None:
        """Run count deterministic steps, advancing `now` by dt seconds each step.

        Args:
            count: Number of steps to run.
            dt: Seconds to advance `now` per step.
        """
        now = 0.0
        for _ in range(count):
            now += dt
            self.step(now)
```

- [ ] **Step 4: Update the sil package exports**

Overwrite `packages/sim/src/sim/sil/__init__.py`:

```python
"""SIL harness: run the real flight apps over sim drivers and step them deterministically."""

from sim.sil.runner import SilHarness, SilSystem, build_sil_system

__all__ = ["SilHarness", "SilSystem", "build_sil_system"]
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest packages/sim/tests/test_sil_closed_loop.py -v`
Expected: 2 passed. (Nominal: payload reaches TRACKING by frame 3 and commands at now=3/5/7 under the 0.5 Hz rate limit, moving the SimGimbal; heartbeats keep the watchdog satisfied so no SAFE. Thermal: the 95.0 reading at step 3 exceeds the 80.0 C limit -> THERMAL_OVER_LIMIT -> FDIR publishes SAFE.)

If the nominal test sees an unexpected SAFE, confirm the harness publishes heartbeats for every MONITORED_SUBSYSTEMS entry BEFORE `fault.tick`. If no gimbal command appears, confirm `dt=1.0` (the 0.5 Hz rate limit needs `now` to advance >= 2.0 s between commands).

- [ ] **Step 6: Commit**

```bash
git add packages/sim/src/sim/sil/runner.py packages/sim/src/sim/sil/__init__.py packages/sim/tests/test_sil_closed_loop.py
git commit -m "feat(sim): add SIL harness + closed-loop integration test"
```

---

### Task 3: Full gate sweep

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
- `mypy packages` -> Success (all green; ~117 source files).
- `lint-imports` -> Contracts: 7 kept, 0 broken. (`sim` importing `flight.*` including `flight.hal.drivers_sim` is allowed; no contract constrains `sim -> flight`.)
- `pytest packages -m "not e2e"` -> 178 passed, 1 skipped (174 + 4 new: 2 scene + 2 SIL).

- [ ] **Step 2: Commit any formatting fix (only if Step 1 required one)**

```bash
git add packages/sim
git commit -m "style: ruff-format SIL harness"
```

---

## HARD RULES for the implementer

- Touch ONLY the files named in Tasks 1-2 (the `sim/scene` + `sim/sil` modules/inits and the two sim test files).
- Do NOT modify `src/pact/**` or any `flight/**` source (this phase only consumes the flight API via `build_apps` and the sim drivers). Do NOT build `sim/twin` (out of scope). Do NOT stage the pre-existing dirty working-tree entries (`src/pact/fault/detector.py`, `tests/**`, `.idea/*`, `.claude/settings.local.json`, `.coverage`, `bash.exe.stackdump`).
- Commits are LOCAL only; do not push.
- Do NOT mark the SIL test `e2e` — it must run in the default `pytest packages -m "not e2e"` gate as the integration proof.
- PowerShell/Windows: `uv run ...` for all gates; `git -m` single-quoted strings (no here-strings).
- Python 3.14 / PEP 758: never add parens to except clauses. Use `from __future__ import annotations`.
- If mypy reports `no-any-return`, assign to a locally-annotated variable first — never `# type: ignore`. If a gate fails, fix the cause; never weaken a test assertion.

## Self-Review (spec coverage)

- SIL runs the real flight apps over sim drivers via the shared `build_apps`. ✓ Task 2 (`build_sil_system`).
- Closed-loop data path proven: plume -> detection -> gimbal command + telemetry. ✓ Task 2 (nominal test).
- Closed-loop FDIR path proven: thermal over-limit -> SAFE. ✓ Task 2 (thermal-fault test).
- Deterministic, single-threaded, runs in CI (not `e2e`). ✓.
- Scene generation isolated and reusable. ✓ Task 1.
- `sim/twin` deferred. ✓ HARD RULES.
```
