# Phase 6: Fault (FDIR) Subsystem Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the fault-detection/isolation/recovery (FDIR) subsystem into `flight/fault`: a pure heartbeat watchdog, a pure fault-to-mode policy (replacing the legacy dynamic-dispatch handler table), and a bus-wired `FaultApp` that consumes heartbeats + fault events and publishes safe-mode transitions.

**Architecture:** Pure core + thin shell, matching the payload subsystem. `watchdog.py` holds `WatchdogEntry` + `check_heartbeats` (pure miss-counting that emits `WATCHDOG_EXPIRE`). `policy.py` holds `SAFE_TRIGGERING_FAULTS` (a `frozenset[FaultCode]`) + pure `decide_mode_change`/`enter_safe_mode`/`exit_safe_mode`. `app.py` holds `FaultApp`: it subscribes to `HeartbeatMsg` and `FaultEventMsg` on the `MessageBus`, runs the watchdog each tick, applies the policy, and publishes `ModeChangeMsg`. All time is injected via `Clock`.

**Tech Stack:** Python 3.14, frozen dataclasses, typed `MessageBus`, injected `Clock`, `Result`-free (this subsystem produces messages, not Results). mypy --strict, ruff (line-length 100), import-linter, pytest.

---

## Context the implementer needs

**Legacy source being migrated (faithful where it matters):**
- `src/pact/fault/watchdog.py` — `WatchdogEntry` + `check_heartbeats(entries, now, max_miss_count) -> (updated, faults)`. Migrate the miss-counting logic verbatim; inject the fault timestamp instead of calling `datetime.now` internally.
- `src/pact/fault/handlers.py` — the `FAULT_HANDLERS` dict mapping each `FaultCode` to a handler returning `ModeChangeMsg | None`. The handlers that return SAFE: `INFERENCE_NAN`, `CAMERA_STALL`, `THERMAL_OVER_LIMIT`, `POWER_OVER_LIMIT`, `GIMBAL_RUNAWAY`, `WATCHDOG_EXPIRE`, `MODEL_CORRUPT`, `PROCESS_DIED`. The handlers that return None (log-and-continue): `NONE`, `INFERENCE_TIMEOUT`, `STORAGE_FULL`, `COMM_TIMEOUT`. This exact partition is preserved by `SAFE_TRIGGERING_FAULTS`.
- `src/pact/fault/safe_mode.py` — `enter_safe_mode`/`exit_safe_mode` (pure ModeChangeMsg builders). Migrate, injecting the timestamp.
- `src/pact/fault/process.py` — the loop (drain heartbeats -> drain faults -> watchdog -> publish mode changes). Reproduced by `FaultApp.tick`/`run` over the bus.

**Deliberate scope decisions (do NOT deviate):**
1. **Replace dynamic dispatch.** Do NOT migrate the `FAULT_HANDLERS` Callable table (it is dynamic dispatch, disallowed by the restructure typing rules). Replace it with `SAFE_TRIGGERING_FAULTS: frozenset[FaultCode]` + a pure `decide_mode_change` membership test. This preserves the exact set of faults that trigger SAFE.
2. **Defer sensor checks.** Do NOT migrate `check_thermal`, `check_power`, or `detect_faults` in this phase. In the subsystem-app/bus architecture each producing subsystem self-reports its faults (the payload app already emits `INFERENCE_NAN`); thermal/power checks belong to the thermal/electrical subsystem phases. Note this in the module docstring.
3. **Clock injection.** Pure functions that need a timestamp take a `now_iso: str` parameter; the app supplies `clock.wall_clock_iso()`. Monotonic time for watchdog intervals is `clock.monotonic_s()`. The legacy `datetime.now(...)` calls are replaced by injected timestamps (behavior-preserving, testable).
4. **Monitored subsystems are injected.** `FaultApp.from_config(cfg, bus, clock, monitored)` takes the tuple of subsystem names to watch. For now the composition root will pass `("payload",)` (the only heartbeat producer built so far); the tuple grows as subsystems are added. Do NOT hardcode the legacy `("imaging", "inference", ...)` list.

**Verified flight signatures this phase depends on:**
- `flight.libs.messages` — `FaultEventMsg(msg_type, timestamp_utc, fault_code, subsystem, detail)`; `HeartbeatMsg(msg_type, timestamp_utc, subsystem, sequence)`; `ModeChangeMsg(msg_type, timestamp_utc, new_mode, requested_by)`.
- `flight.libs.types` — `FaultCode` (members incl. `WATCHDOG_EXPIRE`, `PROCESS_DIED`, `INFERENCE_NAN`, `COMM_TIMEOUT`, `STORAGE_FULL`, `NONE`), `MessageType` (`FAULT_EVENT`, `MODE_CHANGE`), `SystemMode` (`SAFE`, `IDLE`).
- `flight.libs.bus` — `MessageBus.subscribe(type[T]) -> Subscription[T]`, `.publish(object)`; `Subscription.empty()`, `.get_nowait()`.
- `flight.libs.time` — `Clock.monotonic_s() -> float`, `.wall_clock_iso() -> str`; `ManualClock`.
- `flight.libs.config` — `PactConfig`, `FaultConfig(watchdog_interval_s=5.0, watchdog_max_miss_count=3, ...)`.

**Layering (import-linter):** `flight.fault` is a subsystem at the `flight.payload | ... | flight.fault` layer. It may import only `flight.libs.*` (and its own `flight.fault.*` submodules). It must NOT import `flight.core`, peer subsystems, or `flight.hal.drivers_*`. The tests may import anything (tests are outside the `flight` package, unanalyzed by import-linter).

---

### Task 1: Pure heartbeat watchdog

**Files:**
- Create: `packages/flight/src/flight/fault/watchdog.py`
- Test: `packages/flight/tests/test_fault_watchdog.py`

- [ ] **Step 1: Write the failing test**

Create `packages/flight/tests/test_fault_watchdog.py`:

```python
"""Tests for the pure heartbeat watchdog."""

from flight.fault.watchdog import build_entries, check_heartbeats
from flight.libs.types import FaultCode


def test_fresh_entries_have_no_misses() -> None:
    """build_entries starts every subsystem at zero misses."""
    entries = build_entries(("payload",), max_interval_s=5.0, now=100.0)
    assert entries["payload"].miss_count == 0


def test_recent_heartbeat_not_overdue() -> None:
    """A subsystem within max_interval_s is not counted as a miss."""
    entries = build_entries(("payload",), max_interval_s=5.0, now=0.0)
    updated, faults = check_heartbeats(entries, now=3.0, max_miss_count=3, now_iso="t")
    assert updated["payload"].miss_count == 0
    assert faults == []


def test_overdue_increments_miss_without_fault_below_threshold() -> None:
    """One overdue interval increments the miss count but raises no fault yet."""
    entries = build_entries(("payload",), max_interval_s=5.0, now=0.0)
    updated, faults = check_heartbeats(entries, now=6.0, max_miss_count=3, now_iso="t")
    assert updated["payload"].miss_count == 1
    assert faults == []


def test_threshold_emits_watchdog_expire() -> None:
    """Reaching max_miss_count consecutive overdue intervals emits WATCHDOG_EXPIRE."""
    entries = build_entries(("payload",), max_interval_s=5.0, now=0.0)
    now = 0.0
    faults = []
    for _ in range(3):  # max_miss_count = 3
        now += 6.0
        entries, faults = check_heartbeats(entries, now, max_miss_count=3, now_iso="t")
    assert entries["payload"].miss_count == 3
    assert len(faults) == 1
    assert faults[0].fault_code is FaultCode.WATCHDOG_EXPIRE
    assert faults[0].subsystem == "payload"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/flight/tests/test_fault_watchdog.py -v`
Expected: FAIL (ModuleNotFoundError: no module `flight.fault.watchdog`).

- [ ] **Step 3: Write the implementation**

Create `packages/flight/src/flight/fault/watchdog.py`:

```python
"""Heartbeat watchdog: detects silent subsystem death via missed heartbeats.

Pure functions over a dict of WatchdogEntry (one per monitored subsystem). On each
check_heartbeats() call, entries whose last_heartbeat_time is older than max_interval_s
have miss_count incremented; at max_miss_count a FaultEventMsg(WATCHDOG_EXPIRE) is
emitted. The caller owns the clock and the entries dict (state threaded in and out);
this module performs no I/O and reads no clock -- timestamps are injected.

Contains:
  - WatchdogEntry: frozen per-subsystem record (subsystem, last_heartbeat_time in
    monotonic seconds, max_interval_s, miss_count).
  - build_entries: construct the starting entries dict from a tuple of subsystem names.
  - check_heartbeats: increment misses for overdue subsystems and emit WATCHDOG_EXPIRE
    faults at the configured threshold; returns the updated dict and the faults list.

Satisfies: REQ-SAFE-HIGH-002.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass, replace

# internal
from flight.libs.messages import FaultEventMsg
from flight.libs.types import FaultCode, MessageType


@dataclass(frozen=True, slots=True)
class WatchdogEntry:
    """Immutable watchdog record for one monitored subsystem.

    Attributes:
        subsystem: Name matching HeartbeatMsg.subsystem.
        last_heartbeat_time: Monotonic seconds of the most recent received heartbeat.
        max_interval_s: Maximum allowed seconds between heartbeats before a miss counts.
        miss_count: Consecutive overdue intervals since the last received heartbeat.
    """

    subsystem: str
    last_heartbeat_time: float
    max_interval_s: float
    miss_count: int


def build_entries(
    subsystems: tuple[str, ...],
    max_interval_s: float,
    now: float,
) -> dict[str, WatchdogEntry]:
    """Construct the starting watchdog entries dict.

    Each subsystem starts with last_heartbeat_time=now and miss_count=0, giving every
    subsystem a full interval to send its first heartbeat.

    Args:
        subsystems: Names of the subsystems to monitor.
        max_interval_s: Maximum seconds between heartbeats before a miss is counted.
        now: Current monotonic seconds (used as the initial last_heartbeat_time).

    Returns:
        A dict mapping each subsystem name to a fresh WatchdogEntry.
    """
    return {
        name: WatchdogEntry(
            subsystem=name,
            last_heartbeat_time=now,
            max_interval_s=max_interval_s,
            miss_count=0,
        )
        for name in subsystems
    }


def check_heartbeats(
    entries: dict[str, WatchdogEntry],
    now: float,
    max_miss_count: int,
    now_iso: str,
) -> tuple[dict[str, WatchdogEntry], list[FaultEventMsg]]:
    """Increment miss counts for overdue subsystems and emit faults at the threshold.

    For each entry: if (now - last_heartbeat_time) > max_interval_s, increment
    miss_count; if the new miss_count >= max_miss_count, emit a
    FaultEventMsg(WATCHDOG_EXPIRE). Entries that emitted a fault are NOT removed -- the
    caller decides how to respond (e.g. request SAFE mode).

    Args:
        entries: Current watchdog entries (threaded state; not mutated in place).
        now: Current monotonic seconds.
        max_miss_count: Consecutive overdue intervals required to emit WATCHDOG_EXPIRE.
        now_iso: Wall-clock ISO timestamp to stamp on any emitted FaultEventMsg.

    Returns:
        (updated_entries, faults): the new entries dict and any WATCHDOG_EXPIRE faults.
    """
    updated: dict[str, WatchdogEntry] = {}
    faults: list[FaultEventMsg] = []

    for name, entry in entries.items():
        elapsed = now - entry.last_heartbeat_time
        if elapsed > entry.max_interval_s:
            new_miss = entry.miss_count + 1
            updated[name] = replace(entry, miss_count=new_miss)
            if new_miss >= max_miss_count:
                faults.append(
                    FaultEventMsg(
                        msg_type=MessageType.FAULT_EVENT,
                        timestamp_utc=now_iso,
                        fault_code=FaultCode.WATCHDOG_EXPIRE,
                        subsystem=name,
                        detail=(
                            f"watchdog expired: {new_miss} consecutive misses "
                            f"(max_interval_s={entry.max_interval_s})"
                        ),
                    )
                )
        else:
            updated[name] = entry

    return updated, faults
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest packages/flight/tests/test_fault_watchdog.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/flight/src/flight/fault/watchdog.py packages/flight/tests/test_fault_watchdog.py
git commit -m "feat(fault): add pure heartbeat watchdog"
```

---

### Task 2: Pure fault-to-mode policy

**Files:**
- Create: `packages/flight/src/flight/fault/policy.py`
- Test: `packages/flight/tests/test_fault_policy.py`

- [ ] **Step 1: Write the failing test**

Create `packages/flight/tests/test_fault_policy.py`:

```python
"""Tests for the pure fault-to-mode policy."""

from flight.fault.policy import (
    SAFE_TRIGGERING_FAULTS,
    decide_mode_change,
    enter_safe_mode,
    exit_safe_mode,
)
from flight.libs.messages import FaultEventMsg
from flight.libs.types import FaultCode, MessageType, SystemMode


def _fault(code: FaultCode) -> FaultEventMsg:
    """Build a FaultEventMsg carrying the given fault code."""
    return FaultEventMsg(
        msg_type=MessageType.FAULT_EVENT,
        timestamp_utc="t",
        fault_code=code,
        subsystem="payload",
        detail="",
    )


def test_safe_triggering_fault_maps_to_safe() -> None:
    """A SAFE-triggering fault produces a ModeChangeMsg requesting SAFE."""
    change = decide_mode_change(_fault(FaultCode.INFERENCE_NAN), now_iso="t")
    assert change is not None
    assert change.new_mode is SystemMode.SAFE


def test_non_safe_fault_maps_to_none() -> None:
    """Benign faults produce no mode change."""
    assert decide_mode_change(_fault(FaultCode.COMM_TIMEOUT), now_iso="t") is None
    assert decide_mode_change(_fault(FaultCode.STORAGE_FULL), now_iso="t") is None


def test_enter_and_exit_safe_mode() -> None:
    """enter_safe_mode requests SAFE (tagged with the reason); exit requests IDLE."""
    enter = enter_safe_mode(FaultCode.GIMBAL_RUNAWAY, now_iso="t")
    assert enter.new_mode is SystemMode.SAFE
    assert "GIMBAL_RUNAWAY" in enter.requested_by
    leave = exit_safe_mode("ground_cmd", now_iso="t")
    assert leave.new_mode is SystemMode.IDLE
    assert "ground_cmd" in leave.requested_by


def test_safe_triggering_set_membership() -> None:
    """The SAFE-triggering set matches the legacy handler partition."""
    assert FaultCode.PROCESS_DIED in SAFE_TRIGGERING_FAULTS
    assert FaultCode.WATCHDOG_EXPIRE in SAFE_TRIGGERING_FAULTS
    assert FaultCode.NONE not in SAFE_TRIGGERING_FAULTS
    assert FaultCode.INFERENCE_TIMEOUT not in SAFE_TRIGGERING_FAULTS
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/flight/tests/test_fault_policy.py -v`
Expected: FAIL (no module `flight.fault.policy`).

- [ ] **Step 3: Write the implementation**

Create `packages/flight/src/flight/fault/policy.py`:

```python
"""Fault-to-mode policy and SAFE-mode message construction (pure).

Replaces the legacy per-FaultCode Callable dispatch table (FAULT_HANDLERS) with an
explicit, statically typed policy: a frozenset of SAFE-triggering FaultCodes plus a
pure decide_mode_change() that returns a ModeChangeMsg(SAFE) for those codes and None
for the rest. This removes dynamic dispatch (a function-pointer table) in favor of a
direct membership test while preserving the exact partition of faults the legacy
handlers used: SAFE-triggering = {INFERENCE_NAN, CAMERA_STALL, THERMAL_OVER_LIMIT,
POWER_OVER_LIMIT, GIMBAL_RUNAWAY, WATCHDOG_EXPIRE, MODEL_CORRUPT, PROCESS_DIED};
log-and-continue = {NONE, INFERENCE_TIMEOUT, STORAGE_FULL, COMM_TIMEOUT}.

Contains:
  - SAFE_TRIGGERING_FAULTS: the FaultCodes that require a transition to SystemMode.SAFE.
  - enter_safe_mode / exit_safe_mode: build SAFE-entry / SAFE-exit ModeChangeMsg.
  - decide_mode_change: map a FaultEventMsg to a ModeChangeMsg(SAFE) or None.

Satisfies: REQ-SAFE-HIGH-002, REQ-GIMB-HIGH-003.
"""

from __future__ import annotations

from flight.libs.messages import FaultEventMsg, ModeChangeMsg
from flight.libs.types import FaultCode, MessageType, SystemMode

SAFE_TRIGGERING_FAULTS: frozenset[FaultCode] = frozenset(
    {
        FaultCode.INFERENCE_NAN,
        FaultCode.CAMERA_STALL,
        FaultCode.THERMAL_OVER_LIMIT,
        FaultCode.POWER_OVER_LIMIT,
        FaultCode.GIMBAL_RUNAWAY,
        FaultCode.WATCHDOG_EXPIRE,
        FaultCode.MODEL_CORRUPT,
        FaultCode.PROCESS_DIED,
    }
)


def enter_safe_mode(reason: FaultCode, now_iso: str) -> ModeChangeMsg:
    """Build a ModeChangeMsg requesting transition to SystemMode.SAFE.

    Args:
        reason: The FaultCode that triggered SAFE entry; embedded in requested_by.
        now_iso: Wall-clock ISO timestamp for the message.

    Returns:
        A ModeChangeMsg with new_mode=SystemMode.SAFE.
    """
    return ModeChangeMsg(
        msg_type=MessageType.MODE_CHANGE,
        timestamp_utc=now_iso,
        new_mode=SystemMode.SAFE,
        requested_by=f"safe_mode_entry:{reason.value}",
    )


def exit_safe_mode(cleared_by: str, now_iso: str) -> ModeChangeMsg:
    """Build a ModeChangeMsg requesting transition out of SAFE to IDLE.

    SAFE exit requires an explicit ground command; this only constructs the message.

    Args:
        cleared_by: Identifier of the operator/command authorising the exit.
        now_iso: Wall-clock ISO timestamp for the message.

    Returns:
        A ModeChangeMsg with new_mode=SystemMode.IDLE.
    """
    return ModeChangeMsg(
        msg_type=MessageType.MODE_CHANGE,
        timestamp_utc=now_iso,
        new_mode=SystemMode.IDLE,
        requested_by=f"safe_mode_exit:{cleared_by}",
    )


def decide_mode_change(event: FaultEventMsg, now_iso: str) -> ModeChangeMsg | None:
    """Map a fault event to a mode-change request, or None if it is benign.

    Args:
        event: The FaultEventMsg to evaluate.
        now_iso: Wall-clock ISO timestamp for any produced message.

    Returns:
        A ModeChangeMsg(SAFE) if event.fault_code is in SAFE_TRIGGERING_FAULTS, else None.
    """
    if event.fault_code in SAFE_TRIGGERING_FAULTS:
        return enter_safe_mode(event.fault_code, now_iso)
    return None
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest packages/flight/tests/test_fault_policy.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/flight/src/flight/fault/policy.py packages/flight/tests/test_fault_policy.py
git commit -m "feat(fault): add pure fault-to-mode policy (replaces dispatch table)"
```

---

### Task 3: FaultApp bus shell

**Files:**
- Create: `packages/flight/src/flight/fault/app.py`
- Test: `packages/flight/tests/test_fault_app.py`

- [ ] **Step 1: Write the failing test**

Create `packages/flight/tests/test_fault_app.py`:

```python
"""Integration tests for the fault subsystem app (watchdog + fault routing over the bus)."""

from flight.fault.app import FaultApp
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.messages import FaultEventMsg, HeartbeatMsg, ModeChangeMsg
from flight.libs.time import ManualClock
from flight.libs.types import FaultCode, MessageType, SystemMode


def _heartbeat(subsystem: str, seq: int) -> HeartbeatMsg:
    """Build a HeartbeatMsg for the given subsystem and sequence number."""
    return HeartbeatMsg(
        msg_type=MessageType.HEARTBEAT,
        timestamp_utc="t",
        subsystem=subsystem,
        sequence=seq,
    )


def _fault(code: FaultCode) -> FaultEventMsg:
    """Build a FaultEventMsg carrying the given fault code from the payload subsystem."""
    return FaultEventMsg(
        msg_type=MessageType.FAULT_EVENT,
        timestamp_utc="t",
        fault_code=code,
        subsystem="payload",
        detail="",
    )


def _app() -> tuple[FaultApp, MessageBus]:
    """Assemble a FaultApp monitoring 'payload' over a fresh bus and ManualClock."""
    bus = MessageBus()
    app = FaultApp.from_config(PactConfig(), bus, ManualClock(), ("payload",))
    return app, bus


def test_heartbeats_keep_subsystem_alive() -> None:
    """A subsystem that keeps sending heartbeats never trips the watchdog."""
    app, bus = _app()
    mode_sub = bus.subscribe(ModeChangeMsg)
    entries = app.initial_entries()
    now = 0.0
    for seq in range(5):
        now += 5.0
        bus.publish(_heartbeat("payload", seq))
        entries = app.tick(entries, now)
    assert entries["payload"].miss_count == 0
    assert mode_sub.empty()


def test_silent_subsystem_triggers_safe() -> None:
    """A subsystem that stops sending heartbeats trips the watchdog into SAFE."""
    app, bus = _app()
    mode_sub = bus.subscribe(ModeChangeMsg)
    entries = app.initial_entries()
    now = 0.0
    for _ in range(3):  # watchdog_max_miss_count = 3
        now += 10.0  # > watchdog_interval_s (5.0) each tick, no heartbeats published
        entries = app.tick(entries, now)
    assert not mode_sub.empty()
    assert mode_sub.get_nowait().new_mode is SystemMode.SAFE


def test_fault_event_routed_to_safe() -> None:
    """A SAFE-triggering FaultEventMsg on the bus is routed to a ModeChangeMsg(SAFE)."""
    app, bus = _app()
    mode_sub = bus.subscribe(ModeChangeMsg)
    entries = app.initial_entries()
    bus.publish(_fault(FaultCode.PROCESS_DIED))
    app.tick(entries, now=1.0)
    assert not mode_sub.empty()
    assert mode_sub.get_nowait().new_mode is SystemMode.SAFE


def test_benign_fault_not_routed() -> None:
    """A non-SAFE fault (COMM_TIMEOUT) produces no mode change."""
    app, bus = _app()
    mode_sub = bus.subscribe(ModeChangeMsg)
    entries = app.initial_entries()
    bus.publish(_fault(FaultCode.COMM_TIMEOUT))
    app.tick(entries, now=1.0)
    assert mode_sub.empty()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/flight/tests/test_fault_app.py -v`
Expected: FAIL (no module `flight.fault.app`).

- [ ] **Step 3: Write the implementation**

Create `packages/flight/src/flight/fault/app.py`:

```python
"""Fault subsystem app: heartbeat watchdog + fault-to-mode router over the bus.

Subscribes to HeartbeatMsg and FaultEventMsg from every subsystem, runs the pure
watchdog each tick, applies the SAFE-mode policy, and publishes ModeChangeMsg. The
imperative shell owns the bus subscriptions, the clock, and the watchdog-entry dict;
all decision logic is pure (watchdog.check_heartbeats, policy.decide_mode_change).

Contains:
  - FaultApp: frozen holder of config/bus/clock/subscriptions. from_config() subscribes
    to the bus; initial_entries() seeds the watchdog dict; tick() runs one
    drain-heartbeats -> route-faults -> watchdog cycle (threading the entries dict and
    publishing any ModeChangeMsg); run() is the periodic loop.

Non-obvious notes:
  - The arbiter/watchdog interval time is Clock.monotonic_s(); message timestamps use
    Clock.wall_clock_iso(). tick() takes `now` explicitly so it is deterministic in tests.
  - Thermal/power/inference-latency self-checks live in their producing subsystems, not
    here; this app only watches heartbeats and routes already-raised FaultEventMsgs.

Satisfies: REQ-SAFE-HIGH-002, REQ-OPER-HIGH-002.
"""

from __future__ import annotations

# stdlib
import threading
from dataclasses import dataclass, replace

# internal
from flight.fault.policy import decide_mode_change
from flight.fault.watchdog import WatchdogEntry, build_entries, check_heartbeats
from flight.libs.bus import MessageBus, Subscription
from flight.libs.config import FaultConfig, PactConfig
from flight.libs.messages import FaultEventMsg, HeartbeatMsg
from flight.libs.time import Clock


@dataclass(frozen=True)
class FaultApp:
    """FDIR subsystem app: heartbeat watchdog and fault-to-mode router over the bus.

    Frozen to prevent field reassignment; the held bus/clock/subscriptions are mutable
    services injected by the composition root.
    """

    cfg: FaultConfig
    bus: MessageBus
    clock: Clock
    monitored: tuple[str, ...]
    heartbeats: Subscription[HeartbeatMsg]
    faults: Subscription[FaultEventMsg]

    @staticmethod
    def from_config(
        cfg: PactConfig,
        bus: MessageBus,
        clock: Clock,
        monitored: tuple[str, ...],
    ) -> FaultApp:
        """Assemble a FaultApp and subscribe it to heartbeats and fault events.

        Args:
            cfg: Top-level PactConfig (cfg.fault is retained).
            bus: The MessageBus to subscribe to and publish onto.
            clock: Injected Clock.
            monitored: Names of the subsystems whose heartbeats are watched.

        Returns:
            A FaultApp holding fresh HeartbeatMsg and FaultEventMsg subscriptions.
        """
        return FaultApp(
            cfg=cfg.fault,
            bus=bus,
            clock=clock,
            monitored=monitored,
            heartbeats=bus.subscribe(HeartbeatMsg),
            faults=bus.subscribe(FaultEventMsg),
        )

    def initial_entries(self) -> dict[str, WatchdogEntry]:
        """Seed the watchdog entries dict for all monitored subsystems at the current time."""
        return build_entries(
            self.monitored, self.cfg.watchdog_interval_s, self.clock.monotonic_s()
        )

    def tick(self, entries: dict[str, WatchdogEntry], now: float) -> dict[str, WatchdogEntry]:
        """Run one watchdog + fault-routing cycle, publishing any mode changes.

        Drains all pending heartbeats (resetting each known subsystem's miss count),
        routes all pending fault events through the SAFE-mode policy, then runs the
        watchdog and routes any WATCHDOG_EXPIRE faults. Every resulting ModeChangeMsg
        is published to the bus.

        Args:
            entries: Current watchdog entries (threaded state; not mutated in place).
            now: Current monotonic seconds.

        Returns:
            The updated watchdog entries dict.
        """
        working = dict(entries)

        while not self.heartbeats.empty():
            heartbeat = self.heartbeats.get_nowait()
            if heartbeat.subsystem in working:
                working[heartbeat.subsystem] = replace(
                    working[heartbeat.subsystem], last_heartbeat_time=now, miss_count=0
                )

        while not self.faults.empty():
            event = self.faults.get_nowait()
            change = decide_mode_change(event, self.clock.wall_clock_iso())
            if change is not None:
                self.bus.publish(change)

        updated, watchdog_faults = check_heartbeats(
            working, now, self.cfg.watchdog_max_miss_count, self.clock.wall_clock_iso()
        )
        for fault in watchdog_faults:
            change = decide_mode_change(fault, self.clock.wall_clock_iso())
            if change is not None:
                self.bus.publish(change)

        return updated

    def run(self, stop_event: threading.Event) -> None:
        """Run the FDIR loop until stop_event is set.

        Seeds the watchdog entries, then ticks every cfg.watchdog_interval_s seconds.
        Uses stop_event.wait(timeout=...) so shutdown is immediate.

        Args:
            stop_event: threading.Event; the loop exits cleanly once it is set.
        """
        entries = self.initial_entries()
        while not stop_event.is_set():
            now = self.clock.monotonic_s()
            entries = self.tick(entries, now)
            stop_event.wait(timeout=self.cfg.watchdog_interval_s)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest packages/flight/tests/test_fault_app.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/flight/src/flight/fault/app.py packages/flight/tests/test_fault_app.py
git commit -m "feat(fault): add FaultApp bus shell (watchdog + fault routing)"
```

---

### Task 4: Package exports + full gate sweep

**Files:**
- Modify: `packages/flight/src/flight/fault/__init__.py`

- [ ] **Step 1: Write the package exports**

Overwrite `packages/flight/src/flight/fault/__init__.py` with:

```python
"""Fault (FDIR) subsystem: heartbeat watchdog, fault-to-mode policy, and the FDIR app."""

from flight.fault.app import FaultApp
from flight.fault.policy import (
    SAFE_TRIGGERING_FAULTS,
    decide_mode_change,
    enter_safe_mode,
    exit_safe_mode,
)
from flight.fault.watchdog import WatchdogEntry, build_entries, check_heartbeats

__all__ = [
    "SAFE_TRIGGERING_FAULTS",
    "FaultApp",
    "WatchdogEntry",
    "build_entries",
    "check_heartbeats",
    "decide_mode_change",
    "enter_safe_mode",
    "exit_safe_mode",
]
```

- [ ] **Step 2: Run every CI gate, scoped to packages/**

```bash
uv run ruff check packages
uv run ruff format --check packages
uv run mypy packages
uv run lint-imports
uv run pytest packages -m "not e2e"
```

Expected:
- `ruff check packages` -> All checks passed!
- `ruff format --check packages` -> all files already formatted. If `watchdog.py`, `policy.py`, `app.py`, or any new test would be reformatted, run `uv run ruff format packages` and commit with `style: ruff-format fault subsystem`.
- `mypy packages` -> Success (now 91 source files: 88 + watchdog + policy + app).
- `lint-imports` -> Contracts: 7 kept, 0 broken. (Confirms `flight.fault` imports only `flight.libs.*`.)
- `pytest packages -m "not e2e"` -> 145 passed, 1 skipped (133 + 12 new fault tests).

- [ ] **Step 3: Commit the exports (and any formatting fix)**

```bash
git add packages/flight/src/flight/fault/__init__.py
git commit -m "feat(fault): export fault subsystem public API"
```

---

## HARD RULES for the implementer

- Touch ONLY: `packages/flight/src/flight/fault/{watchdog,policy,app,__init__}.py` and `packages/flight/tests/test_fault_{watchdog,policy,app}.py`.
- Do NOT modify `src/pact/**` (additive migration; `src/pact` stays untouched). Do NOT stage the pre-existing dirty working-tree entries (`src/pact/fault/detector.py`, `tests/**`, `.idea/*`, `.claude/settings.local.json`, `.coverage`, `bash.exe.stackdump`).
- Commits are LOCAL only; do not push.
- Do NOT migrate `FAULT_HANDLERS` (dynamic dispatch), `check_thermal`, `check_power`, or `detect_faults` — they are deliberately out of scope per the policy/defer decisions above.
- PowerShell/Windows: use `uv run ...` for all gates; use `git -m` with single-quoted strings (no here-strings).
- Python 3.14 / PEP 758: `except A, B:` without parens is valid and is the ruff-format-normalized form — never add parens. Use `from __future__ import annotations` so `-> FaultApp` resolves unquoted.
- If a gate fails, fix the cause; never weaken a test assertion or add `# type: ignore`.

## Self-Review (spec coverage)

- Watchdog (heartbeat miss-counting -> WATCHDOG_EXPIRE). ✓ Task 1.
- Fault-to-mode policy (SAFE partition preserved, no dynamic dispatch). ✓ Task 2.
- Safe-mode enter/exit message builders. ✓ Task 2.
- Bus-wired FDIR app (consume HeartbeatMsg + FaultEventMsg, publish ModeChangeMsg). ✓ Task 3.
- Layering preserved (fault depends only on libs). ✓ Task 4 `lint-imports`.
- Clock injection (no `datetime.now`/`time` inside the migrated logic). ✓ all tasks.
```
