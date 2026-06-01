# Phase 4 -- Core Foundations (Clock, Bus, config_loader) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the app-independent core foundations the subsystem apps depend on: `flight.libs.time` (an injectable `Clock`), `flight.libs.bus` (a type-routed in-process pub/sub `MessageBus`), and `flight.core.config_loader` (migrated `load_config` returning `Result[PactConfig, str]`) -- all `mypy --strict` and `ruff` clean, all gates green.

**Architecture:** `Clock` is a `Protocol` with `monotonic_s()` (control intervals, timeouts, rate limits) and `wall_clock_iso()` (message timestamps), mirroring the existing monotonic-vs-wall-clock split; `RealClock` uses the system clocks and `ManualClock` is deterministic for tests. `MessageBus` generalizes the current per-channel queue pattern into one bus routed by exact message type: `publish(msg)` delivers to every `Subscription` registered for `type(msg)`; transport is in-process `queue.Queue` (what unit tests and single-process SIL use), swappable later without changing the API. `config_loader` is migrated verbatim (import paths only) from `src/pact/ops/config_loader.py` into `flight/core`. The composition root that wires these into running apps is intentionally deferred until apps exist.

**Tech Stack:** Python 3.14, typing.Protocol, generics, queue.Queue, tomllib, frozen dataclasses, pytest, mypy --strict, ruff, import-linter.

---

## Context for the implementer

- `src/pact/ops/config_loader.py` is the migration source for Task 3. READ it and reproduce `load_config`, `_deep_merge`, `_validate`, `_build_pact_config` (and any helpers) faithfully, changing ONLY imports: `PactConfig` and the six sub-config dataclasses come from `flight.libs.config`; `Result`/`Ok`/`Err` from `flight.libs.types`. It returns `Result[PactConfig, str]` and deep-merges `config/default.toml` with an optional override (`config/flight.toml`).
- The new `Clock` and `MessageBus` are greenfield; their full code is given below.
- `flight.core` and `flight.libs` package dirs already exist (empty `__init__.py` from Phase 1). `flight.libs.bus` and `flight.libs.time` are NEW submodules. `flight.core.config_loader` is a NEW module in the existing `flight.core` package.
- MUST pass `uv run mypy packages` (strict) and `uv run ruff check packages`. Do NOT modify `src/pact/`. Do NOT build the composition root / app spawning. Stage only named files. Commit locally; no push. ASCII only. New test functions annotated `-> None`.
- Layering: `flight.libs.time` and `flight.libs.bus` import nothing from other libs (self-contained); `flight.core.config_loader` imports `flight.libs.{config,types}` (core is above libs -- allowed). No `.importlinter` change is required this phase.

## File structure (created in this phase)

```
packages/flight/src/flight/libs/time/__init__.py        # re-export Clock, RealClock, ManualClock
packages/flight/src/flight/libs/time/clock.py           # NEW
packages/flight/src/flight/libs/bus/__init__.py         # re-export MessageBus, Subscription
packages/flight/src/flight/libs/bus/bus.py              # NEW
packages/flight/src/flight/core/config_loader.py        # migrated from src/pact/ops/config_loader.py
packages/flight/src/flight/core/__init__.py             # MODIFY: re-export load_config
packages/flight/tests/test_clock.py                     # NEW
packages/flight/tests/test_bus.py                       # NEW
packages/flight/tests/test_config_loader.py             # NEW
```

---

## Task 1: `flight.libs.time` -- the Clock abstraction

**Files:**
- Create: `packages/flight/src/flight/libs/time/clock.py`
- Create: `packages/flight/src/flight/libs/time/__init__.py`
- Test: `packages/flight/tests/test_clock.py`

- [ ] **Step 1: Create `clock.py`**

```python
"""Clock abstraction: time is injected, never read inside pure logic.

Separates monotonic time (control intervals, timeouts, rate limits) from wall-clock
time (message timestamps), mirroring how the existing code sources time. Pure
functions and app shells receive a Clock; the composition root owns the concrete
instance. ManualClock makes time deterministic and advanceable in tests.
"""

import time as _time
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Injected time source."""

    def monotonic_s(self) -> float:
        """Monotonic seconds since an arbitrary epoch (intervals, timeouts, rates)."""
        ...

    def wall_clock_iso(self) -> str:
        """Current UTC time as ISO 8601 with millisecond precision (message stamps)."""
        ...


class RealClock:
    """Production clock backed by time.monotonic() and the system UTC clock."""

    def monotonic_s(self) -> float:
        """Return time.monotonic() in seconds."""
        return _time.monotonic()

    def wall_clock_iso(self) -> str:
        """Return current UTC time as 'YYYY-MM-DDTHH:MM:SS.mmmZ'."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class ManualClock:
    """Deterministic clock for tests; monotonic time is advanced explicitly."""

    def __init__(
        self,
        monotonic_s: float = 0.0,
        wall_clock: str = "2026-01-01T00:00:00.000Z",
    ) -> None:
        """Initialize the manual clock.

        Args:
            monotonic_s: Initial monotonic seconds.
            wall_clock: Initial wall-clock ISO 8601 string.
        """
        self._monotonic_s = monotonic_s
        self._wall_clock = wall_clock

    def monotonic_s(self) -> float:
        """Return the current (manually set) monotonic seconds."""
        return self._monotonic_s

    def wall_clock_iso(self) -> str:
        """Return the current (manually set) wall-clock ISO string."""
        return self._wall_clock

    def advance(self, delta_s: float) -> None:
        """Advance monotonic time by delta_s seconds."""
        self._monotonic_s += delta_s

    def set_wall_clock(self, wall_clock: str) -> None:
        """Set the wall-clock ISO string returned by wall_clock_iso()."""
        self._wall_clock = wall_clock
```

- [ ] **Step 2: Create `time/__init__.py`**

```python
"""Injectable clock abstraction (monotonic + wall-clock)."""

from flight.libs.time.clock import Clock, ManualClock, RealClock

__all__ = ["Clock", "ManualClock", "RealClock"]
```

- [ ] **Step 3: Write `test_clock.py`**

```python
"""Tests for the Clock abstraction."""

import re

from flight.libs.time import Clock, ManualClock, RealClock


def test_real_clock_monotonic_non_decreasing() -> None:
    """RealClock.monotonic_s is non-decreasing across calls."""
    clock = RealClock()
    first = clock.monotonic_s()
    second = clock.monotonic_s()
    assert second >= first


def test_real_clock_wall_clock_format() -> None:
    """RealClock.wall_clock_iso returns a millisecond ISO 8601 UTC string."""
    stamp = RealClock().wall_clock_iso()
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$", stamp)


def test_manual_clock_advances() -> None:
    """ManualClock advances monotonic time only when told to."""
    clock = ManualClock(monotonic_s=10.0)
    assert clock.monotonic_s() == 10.0
    clock.advance(2.5)
    assert clock.monotonic_s() == 12.5


def test_manual_clock_wall_clock_settable() -> None:
    """ManualClock wall clock is fixed until explicitly set."""
    clock = ManualClock(wall_clock="2026-05-31T00:00:00.000Z")
    assert clock.wall_clock_iso() == "2026-05-31T00:00:00.000Z"
    clock.set_wall_clock("2026-06-01T00:00:00.000Z")
    assert clock.wall_clock_iso() == "2026-06-01T00:00:00.000Z"


def test_clocks_satisfy_protocol() -> None:
    """Both clocks conform to the Clock protocol (typed + runtime)."""
    real: Clock = RealClock()
    manual: Clock = ManualClock()
    assert isinstance(real, Clock)
    assert isinstance(manual, Clock)
```

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest packages/flight/tests/test_clock.py -v` -> PASS. `uv run mypy packages` -> Success. `uv run ruff check packages` -> passed.
```bash
git add packages/flight/src/flight/libs/time packages/flight/tests/test_clock.py
git commit -m "feat(libs): add Clock abstraction (RealClock, ManualClock)"
```

---

## Task 2: `flight.libs.bus` -- the typed pub/sub bus

**Files:**
- Create: `packages/flight/src/flight/libs/bus/bus.py`
- Create: `packages/flight/src/flight/libs/bus/__init__.py`
- Test: `packages/flight/tests/test_bus.py`

- [ ] **Step 1: Create `bus.py`**

```python
"""In-process typed pub/sub message bus.

Generalizes the per-channel queue pattern into one bus routed by exact message type:
publish(msg) delivers msg to every Subscription registered for type(msg). The
composition root owns the bus and injects Subscriptions into apps; apps never
construct queues themselves. Transport is in-process queue.Queue (what unit tests
and single-process SIL use); a multiprocessing-backed transport can replace the
queue factory later without changing this API.
"""

import threading
from queue import Queue
from typing import Generic, TypeVar, cast

_T = TypeVar("_T")


class Subscription(Generic[_T]):
    """A typed receive handle for one subscribed message type."""

    def __init__(self, queue: "Queue[_T]") -> None:
        """Wrap the backing queue for a single subscription."""
        self._queue = queue

    def get(self, timeout: float | None = None) -> _T:
        """Block for the next message, optionally up to timeout seconds.

        Raises:
            queue.Empty: If timeout elapses with no message.
        """
        return self._queue.get(timeout=timeout)

    def get_nowait(self) -> _T:
        """Return the next message immediately.

        Raises:
            queue.Empty: If no message is queued.
        """
        return self._queue.get_nowait()

    def empty(self) -> bool:
        """Return True if no message is currently queued."""
        return self._queue.empty()


class MessageBus:
    """Typed pub/sub bus routed by exact message type (in-process)."""

    def __init__(self, maxsize: int = 0) -> None:
        """Create an empty bus.

        Args:
            maxsize: Per-subscription queue bound (0 = unbounded).
        """
        self._maxsize = maxsize
        self._subscribers: dict[type, list[Queue[object]]] = {}
        self._lock = threading.Lock()

    def subscribe(self, message_type: type[_T]) -> Subscription[_T]:
        """Register interest in a message type and return a receive handle."""
        queue: Queue[object] = Queue(maxsize=self._maxsize)
        with self._lock:
            self._subscribers.setdefault(message_type, []).append(queue)
        return Subscription(cast("Queue[_T]", queue))

    def publish(self, message: object) -> None:
        """Deliver message to every Subscription registered for its exact type."""
        with self._lock:
            queues = list(self._subscribers.get(type(message), []))
        for queue in queues:
            queue.put(message)
```

- [ ] **Step 2: Create `bus/__init__.py`**

```python
"""Typed in-process pub/sub message bus."""

from flight.libs.bus.bus import MessageBus, Subscription

__all__ = ["MessageBus", "Subscription"]
```

- [ ] **Step 3: Write `test_bus.py`**

```python
"""Tests for the typed pub/sub message bus."""

from queue import Empty

import pytest

from flight.libs.bus import MessageBus, Subscription
from flight.libs.messages import HeartbeatMsg, utc_now_iso
from flight.libs.types import MessageType


def _heartbeat(sequence: int) -> HeartbeatMsg:
    """Build a HeartbeatMsg with the given sequence number."""
    return HeartbeatMsg(
        msg_type=MessageType.HEARTBEAT,
        timestamp_utc=utc_now_iso(),
        subsystem="test",
        sequence=sequence,
    )


def test_publish_delivers_to_subscriber() -> None:
    """A subscriber receives a message published for its type."""
    bus = MessageBus()
    sub: Subscription[HeartbeatMsg] = bus.subscribe(HeartbeatMsg)
    bus.publish(_heartbeat(1))
    received = sub.get_nowait()
    assert received.sequence == 1


def test_multiple_subscribers_each_receive() -> None:
    """Every subscriber of a type receives each published message (fan-out)."""
    bus = MessageBus()
    sub_a: Subscription[HeartbeatMsg] = bus.subscribe(HeartbeatMsg)
    sub_b: Subscription[HeartbeatMsg] = bus.subscribe(HeartbeatMsg)
    bus.publish(_heartbeat(7))
    assert sub_a.get_nowait().sequence == 7
    assert sub_b.get_nowait().sequence == 7


def test_no_subscribers_is_noop() -> None:
    """Publishing a type with no subscribers does not raise."""
    bus = MessageBus()
    bus.publish(_heartbeat(1))  # no subscribers; must not raise


def test_subscriber_only_gets_its_type() -> None:
    """A subscription receives only messages of its registered type."""
    bus = MessageBus()
    sub: Subscription[HeartbeatMsg] = bus.subscribe(HeartbeatMsg)
    bus.publish("not a heartbeat")
    assert sub.empty()
    with pytest.raises(Empty):
        sub.get_nowait()
```

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest packages/flight/tests/test_bus.py -v` -> PASS. `uv run mypy packages` -> Success. `uv run ruff check packages` -> passed. `uv run lint-imports` -> 7 contracts kept (libs.bus self-contained, no new contract needed).
```bash
git add packages/flight/src/flight/libs/bus packages/flight/tests/test_bus.py
git commit -m "feat(libs): add typed pub/sub MessageBus"
```

---

## Task 3: `flight.core.config_loader` -- migrate the loader

**Files:**
- Create: `packages/flight/src/flight/core/config_loader.py`
- Modify: `packages/flight/src/flight/core/__init__.py`
- Test: `packages/flight/tests/test_config_loader.py`

- [ ] **Step 1: Migrate `config_loader.py`**

Read `src/pact/ops/config_loader.py`. Reproduce it faithfully into `packages/flight/src/flight/core/config_loader.py`, changing ONLY the imports:
- `from flight.libs.config import (CommsConfig, ControllerConfig, FaultConfig, InferenceConfig, PactConfig, PreprocessingConfig, StorageConfig)` (whichever it uses)
- `from flight.libs.types import Err, Ok, Result`
Keep `load_config`, `_deep_merge`, `_validate`, `_build_pact_config`, and any other helpers exactly as written (same logic, field mapping, validation). Keep the requirement-ID module docstring header. Ensure it passes strict mypy (add minimal annotations only if a faithful copy trips strict mode).

- [ ] **Step 2: Re-export from `core/__init__.py`**

```python
"""Compute / C&DH host (composition root, config loading, scheduling).

This phase populates only config loading; the composition root, scheduler, bus
router wiring, storage, telemetry aggregator, and FDIR coordinator are added as
the subsystem apps come online.
"""

from flight.core.config_loader import load_config

__all__ = ["load_config"]
```

- [ ] **Step 3: Write `test_config_loader.py`**

```python
"""Tests for the migrated config loader."""

from pathlib import Path

from flight.core import load_config
from flight.libs.config import PactConfig
from flight.libs.types import Err, Ok

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_TOML = str(_REPO_ROOT / "config" / "default.toml")
_FLIGHT_TOML = str(_REPO_ROOT / "config" / "flight.toml")


def test_loads_default_config() -> None:
    """load_config returns Ok(PactConfig) for the default TOML."""
    result = load_config(_DEFAULT_TOML)
    assert isinstance(result, Ok)
    assert isinstance(result.value, PactConfig)


def test_flight_override_merges() -> None:
    """The flight.toml override (use_int8 = true) merges over defaults."""
    result = load_config(_DEFAULT_TOML, _FLIGHT_TOML)
    assert isinstance(result, Ok)
    assert result.value.inference.use_int8 is True


def test_missing_file_returns_err() -> None:
    """A missing config path returns Err, not an exception."""
    result = load_config(str(_REPO_ROOT / "config" / "does_not_exist.toml"))
    assert isinstance(result, Err)
```

Note: confirm `load_config`'s signature (positional `config_path`, optional `override_path`) from the source; adjust the calls if the parameter names differ. Confirm `config/flight.toml` sets `use_int8 = true` under `[inference]` (it does, per planning); if the override key differs, assert the actual overridden value instead.

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest packages/flight/tests/test_config_loader.py -v` -> PASS. `uv run mypy packages` -> Success. `uv run ruff check packages` -> passed.
```bash
git add packages/flight/src/flight/core/config_loader.py packages/flight/src/flight/core/__init__.py packages/flight/tests/test_config_loader.py
git commit -m "feat(core): migrate config_loader (load_config -> Result[PactConfig, str])"
```

---

## Task 4: Full gate sweep

**Files:** none (verification)

- [ ] **Step 1: Run every gate exactly as CI does**

```bash
uv run ruff check packages
uv run ruff format --check packages
uv run mypy packages
uv run lint-imports
uv run pytest packages -m "not e2e"
```
Expected: all pass; `lint-imports` reports 7 contracts kept; pytest includes the new clock, bus, and config-loader tests.

- [ ] **Step 2: If `ruff format --check packages` flags new files**, run `uv run ruff format packages`, re-check, and commit:
```bash
git add packages
git commit -m "style: ruff-format new core-foundation files"
```
(Skip if nothing needed reformatting.)

---

## Risks & notes

- **mypy strict + generics:** the bus uses a `TypeVar` and a `cast`; if strict mypy flags the `Queue[object]` -> `Queue[_T]` bridge, keep the `cast` (do not weaken `Subscription`/`publish` signatures). The conformance is that `subscribe(X)` returns `Subscription[X]` whose `get()` yields `X`.
- **Self-contained libs:** `libs.time` and `libs.bus` import nothing from other libs, so no import-linter change is needed. `core.config_loader` importing `libs.{config,types}` is core->libs (allowed by the flight-layers contract).
- **Deferred deliberately:** the composition root, scheduler, bus *router wiring into processes*, storage, telemetry aggregator, FDIR coordinator, and the multiprocessing transport are NOT in this phase. They are built once apps exist (payload onward / SIL-integration). The standardized `CommandMsg` envelope is deferred to the `iss_iface` phase (where command ingest lives).
- **Legacy untouched:** `src/pact/ops/config_loader.py` stays as the legacy loader; the migrated copy in `flight.core` is the go-forward.

## Self-review (performed against the spec)

- **Spec coverage:** `libs/time` Clock (Section 4 clock-ownership; arbiter pattern), `libs/bus` typed pub/sub (Section 5 bus mechanism, transport-agnostic API), `core/config_loader` (Section 9 config distribution). The deferred core services are listed with rationale.
- **Placeholder scan:** no TBD/TODO in plan prose; greenfield code (Clock, MessageBus) given in full; config_loader migration points at the exact source with explicit import-rewrite rules.
- **Type/name consistency:** `Clock`/`RealClock`/`ManualClock`, `MessageBus`/`Subscription`, `load_config` are used identically across modules, `__init__` re-exports, and tests. `monotonic_s()`/`wall_clock_iso()` and `subscribe()`/`publish()`/`get_nowait()` signatures match between definition and test usage.
