# Phase 5c -- Payload Tracking + Gimbal (controller migration) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Faithfully migrate the six pure controller modules into `flight.payload.tracking` (estimation) and `flight.payload.gimbal` (FSM + control law + safety), bringing the existing 60 tests and adding minimal tests for the two untested modules -- `mypy --strict`/`ruff` clean, gates green.

**Architecture:** `flight/payload/tracking/` holds the estimation primitives: `filter` (EMA centroid smoothing), `kalman` (constant-velocity pointing estimator), `tracker` (IoU blob association + persistence). `flight/payload/gimbal/` holds the command path: `arbiter` (the IDLE/ACQUIRING/TRACKING/SCAN/SAFE FSM + `ArbiterState` + `GimbalArbiter`), `lqr` (discrete-LQR control law), `safety` (confidence/area/deadband/rate gates). Every module is a pure leaf that imports only `flight.libs.*`; none import each other (the payload app shell sequences them later, in Phase 5d). This is a behavior-preserving migration -- imports change, algorithms do not.

**Tech Stack:** Python 3.14, numpy, scipy.linalg, frozen dataclasses, pytest, mypy --strict, ruff, import-linter.

---

## Context for the implementer

- Migration sources (READ each; reproduce faithfully, change ONLY imports):
  - `src/pact/controller/filter.py` -- `EmaFilterState`, `ema_update(state, new_centroid, alpha)`. No `pact.*` imports.
  - `src/pact/controller/kalman.py` -- `KalmanState`, `KalmanFilter` (`.from_config(cfg)`, `.initial_state(pan_deg, tilt_deg)`), `predict(kf, state)`, `update(kf, state, observation) -> Ok[KalmanState] | Err[FaultCode]`. Imports `ControllerConfig` from `pact.types.config`, `FaultCode, Ok, Err` from `pact.types.enums`, numpy.
  - `src/pact/controller/tracker.py` -- `compute_iou(box_a, box_b)`, `match_blobs(prev_blobs, new_blobs, iou_threshold)`. Imports `BlobMeta` from `pact.types.messages`.
  - `src/pact/controller/arbiter.py` -- `ArbiterState` (frozen dataclass), `GimbalArbiter` (`__init__(cfg)`, `step(state, result, now)`), and module-private helpers `_any_acquired`, `_select_best_target`, `_rate_ok`. Imports `ControllerConfig` from `pact.types.config`; `GimbalState, MessageType` from `pact.types.enums`; `BlobMeta, GimbalCommandMsg, InferenceResultMsg, TelemetryEventMsg, utc_now_iso` from `pact.types.messages`.
  - `src/pact/controller/lqr.py` -- `LqrController` (`.from_config(cfg)`), `compute_control(controller, state_error)`. Imports `ControllerConfig` from `pact.types.config`, numpy, scipy.linalg.
  - `src/pact/controller/safety.py` -- `apply_confidence_gate`, `apply_min_area_gate`, `check_deadband`, `check_rate_limit`. Imports `FaultCode, Ok, Err, Result` from `pact.types.enums`, `BlobMeta` from `pact.types.messages`.
- Import rewrites (apply to all six modules):
  - `from pact.types.config import ControllerConfig` -> `from flight.libs.config import ControllerConfig`
  - `from pact.types.enums import ...` -> `from flight.libs.types import ...` (GimbalState, MessageType, FaultCode, Ok, Err, Result)
  - `from pact.types.messages import ...` -> `from flight.libs.messages import ...` (BlobMeta, GimbalCommandMsg, InferenceResultMsg, TelemetryEventMsg, utc_now_iso)
- Test sources to ADAPT (copy, retarget imports, keep helpers + assertions identical, annotate `-> None`): `tests/unit/controller/test_filter.py`, `test_tracker.py`, `test_arbiter.py`, `test_safety.py`. `kalman.py` and `lqr.py` have NO existing tests -- write the minimal ones given below (verify against the source).
- `flight.payload` exists (empty `__init__`); `tracking/` and `gimbal/` are new subpackages. `scipy` and `numpy` are already flight deps (added Phase 5b).
- MUST pass `uv run mypy packages` (strict) and `uv run ruff check packages`. Do NOT modify `src/pact/`. Do NOT migrate `process.py` (the orchestration shell -- that is Phase 5d). Stage only named files. Commit locally; no push. ASCII only. No `.importlinter` change (payload -> libs allowed).

## File structure (created in this phase)

```
packages/flight/src/flight/payload/tracking/__init__.py     # re-export filter/kalman/tracker API
packages/flight/src/flight/payload/tracking/filter.py
packages/flight/src/flight/payload/tracking/kalman.py
packages/flight/src/flight/payload/tracking/tracker.py
packages/flight/src/flight/payload/gimbal/__init__.py       # re-export arbiter/lqr/safety API
packages/flight/src/flight/payload/gimbal/arbiter.py
packages/flight/src/flight/payload/gimbal/lqr.py
packages/flight/src/flight/payload/gimbal/safety.py
packages/flight/tests/conftest.py                           # default_config + arbiter_idle_state fixtures
packages/flight/tests/test_filter.py        # adapt
packages/flight/tests/test_tracker.py       # adapt
packages/flight/tests/test_kalman.py        # NEW
packages/flight/tests/test_arbiter.py       # adapt (uses conftest fixtures)
packages/flight/tests/test_safety.py        # adapt
packages/flight/tests/test_lqr.py           # NEW
```

---

## Task 1: `flight.payload.tracking` (filter, kalman, tracker)

**Files:** the 3 tracking modules + `__init__.py` + `test_filter.py`, `test_tracker.py`, `test_kalman.py`.

- [ ] **Step 1: Migrate the three modules** into `flight/payload/tracking/` (faithful copies of `src/pact/controller/{filter,kalman,tracker}.py` with the import rewrites above). Preserve all signatures, the constant-velocity Kalman matrices/logic, the IoU/greedy-match logic, and the requirement-ID docstring headers.

- [ ] **Step 2: Create `tracking/__init__.py`**

```python
"""Payload tracking: target-state estimation and blob association (pure functions).

filter -- EMA centroid smoothing; kalman -- constant-velocity pointing estimator;
tracker -- IoU blob matching and persistence counting.
"""

from flight.payload.tracking.filter import EmaFilterState, ema_update
from flight.payload.tracking.kalman import KalmanFilter, KalmanState, predict, update
from flight.payload.tracking.tracker import compute_iou, match_blobs

__all__ = [
    "EmaFilterState",
    "KalmanFilter",
    "KalmanState",
    "compute_iou",
    "ema_update",
    "match_blobs",
    "predict",
    "update",
]
```

- [ ] **Step 3: Adapt `test_filter.py` and `test_tracker.py`** into `packages/flight/tests/`, retargeting imports to `flight.payload.tracking` and `flight.libs.*`. Keep all local helpers, parametrization, and assertions identical; annotate every test `-> None`.

- [ ] **Step 4: Write `test_kalman.py`** (no source test exists). READ `kalman.py` first and adjust to the real signatures.

```python
"""Tests for the Kalman pointing estimator."""

import numpy as np

from flight.libs.config import ControllerConfig
from flight.libs.types import Ok
from flight.payload.tracking import KalmanFilter, predict, update


def test_predict_keeps_state_shape() -> None:
    """predict returns a KalmanState whose estimate is the 4-vector [pan, tilt, dpan, dtilt]."""
    kf = KalmanFilter.from_config(ControllerConfig())
    state = KalmanFilter.initial_state(pan_deg=0.0, tilt_deg=0.0)
    predicted = predict(kf, state)
    estimate = np.asarray(predicted.x)
    assert estimate.shape == (4,)


def test_update_incorporates_observation() -> None:
    """update returns Ok(KalmanState) for a finite 2D observation."""
    kf = KalmanFilter.from_config(ControllerConfig())
    state = KalmanFilter.initial_state(pan_deg=1.0, tilt_deg=2.0)
    result = update(kf, state, np.array([1.5, 2.5], dtype=np.float64))
    assert isinstance(result, Ok)
    assert np.asarray(result.value.x).shape == (4,)
```

- [ ] **Step 5: Verify and commit**

Run: `uv run pytest packages/flight/tests/test_filter.py packages/flight/tests/test_tracker.py packages/flight/tests/test_kalman.py -v` -> PASS. `uv run mypy packages` -> Success. `uv run ruff check packages` -> passed.
```bash
git add packages/flight/src/flight/payload/tracking packages/flight/tests/test_filter.py packages/flight/tests/test_tracker.py packages/flight/tests/test_kalman.py
git commit -m "feat(payload): migrate tracking estimators (filter, kalman, tracker)"
```

---

## Task 2: `flight.payload.gimbal` (arbiter, lqr, safety) + conftest

**Files:** the 3 gimbal modules + `__init__.py` + `conftest.py` + `test_arbiter.py`, `test_safety.py`, `test_lqr.py`.

- [ ] **Step 1: Migrate the three modules** into `flight/payload/gimbal/` (faithful copies of `src/pact/controller/{arbiter,lqr,safety}.py` with import rewrites). Preserve `ArbiterState`, the FSM transitions and command generation in `GimbalArbiter.step`, the LQR DARE/gain/clamp logic, and all safety gates exactly.

- [ ] **Step 2: Create `gimbal/__init__.py`**

```python
"""Payload gimbal control: the pointing FSM, control law, and safety gates (pure).

arbiter -- the IDLE/ACQUIRING/TRACKING/SCAN/SAFE FSM and command generation;
lqr -- discrete-LQR control law; safety -- confidence/area/deadband/rate gates.
"""

from flight.payload.gimbal.arbiter import ArbiterState, GimbalArbiter
from flight.payload.gimbal.lqr import LqrController, compute_control
from flight.payload.gimbal.safety import (
    apply_confidence_gate,
    apply_min_area_gate,
    check_deadband,
    check_rate_limit,
)

__all__ = [
    "ArbiterState",
    "GimbalArbiter",
    "LqrController",
    "apply_confidence_gate",
    "apply_min_area_gate",
    "check_deadband",
    "check_rate_limit",
    "compute_control",
]
```

- [ ] **Step 3: Create `packages/flight/tests/conftest.py`** (the two fixtures the arbiter tests need)

```python
"""Shared fixtures for the flight test suite."""

from pathlib import Path

import pytest

from flight.core import load_config
from flight.libs.config import PactConfig
from flight.libs.types import GimbalState, Ok
from flight.payload.gimbal import ArbiterState

_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def default_config() -> PactConfig:
    """PactConfig loaded from config/default.toml (frozen; use replace() to modify)."""
    result = load_config(str(_REPO_ROOT / "config" / "default.toml"))
    assert isinstance(result, Ok)
    return result.value


@pytest.fixture
def arbiter_idle_state() -> ArbiterState:
    """An ArbiterState in GimbalState.IDLE with no tracked blobs."""
    return ArbiterState(
        gimbal_state=GimbalState.IDLE,
        tracked_blobs=(),
        idle_duration_s=0.0,
        last_command_time=0.0,
        current_target_id=None,
    )
```

Note: if `ArbiterState` has additional required (non-default) fields beyond these, add them per the source; the source gives `scan_pan_deg` a default, so it may be omitted.

- [ ] **Step 4: Adapt `test_arbiter.py` and `test_safety.py`** into `packages/flight/tests/`, retargeting imports (`flight.payload.gimbal`, `flight.libs.*`). Keep helpers, parametrization, and assertions identical. `test_arbiter.py` uses the `default_config` and `arbiter_idle_state` fixtures from the new conftest -- keep those fixture names.

- [ ] **Step 5: Write `test_lqr.py`** (no source test exists). READ `lqr.py` first and adjust to the real signatures.

```python
"""Tests for the LQR gimbal control law."""

import numpy as np

from flight.libs.config import ControllerConfig
from flight.payload.gimbal import LqrController, compute_control


def test_command_clamped_to_max_slew() -> None:
    """A large pointing error produces a command clamped to +/- max_slew_deg_s."""
    cfg = ControllerConfig()
    controller = LqrController.from_config(cfg)
    command = np.asarray(compute_control(controller, np.array([1000.0, 1000.0, 0.0, 0.0])))
    assert command.shape == (2,)
    assert abs(float(command[0])) <= cfg.max_slew_deg_s + 1e-9
    assert abs(float(command[1])) <= cfg.max_slew_deg_s + 1e-9


def test_zero_error_zero_command() -> None:
    """No pointing error yields an approximately zero slew command."""
    controller = LqrController.from_config(ControllerConfig())
    command = np.asarray(compute_control(controller, np.zeros(4, dtype=np.float64)))
    assert abs(float(command[0])) < 1e-9
    assert abs(float(command[1])) < 1e-9
```

- [ ] **Step 6: Verify and commit**

Run: `uv run pytest packages/flight/tests/test_arbiter.py packages/flight/tests/test_safety.py packages/flight/tests/test_lqr.py -v` -> PASS. `uv run mypy packages` -> Success. `uv run ruff check packages` -> passed.
```bash
git add packages/flight/src/flight/payload/gimbal packages/flight/tests/conftest.py packages/flight/tests/test_arbiter.py packages/flight/tests/test_safety.py packages/flight/tests/test_lqr.py
git commit -m "feat(payload): migrate gimbal arbiter/lqr/safety + flight conftest fixtures"
```

---

## Task 3: Full gate sweep

**Files:** none (verification)

- [ ] **Step 1: Run every gate exactly as CI does**

```bash
uv run ruff check packages
uv run ruff format --check packages
uv run mypy packages
uv run lint-imports
uv run pytest packages -m "not e2e"
```
Expected: all pass; `lint-imports` 7 contracts kept; pytest includes the ~60 migrated controller tests + 4 new kalman/lqr tests.

- [ ] **Step 2: If `ruff format --check packages` flags new files**, run `uv run ruff format packages`, re-check, commit:
```bash
git add packages
git commit -m "style: ruff-format new tracking/gimbal files"
```
(Skip if nothing needed reformatting.)

---

## Risks & notes

- **conftest scope:** `packages/flight/tests/conftest.py` applies to ALL flight tests. The two fixtures are additive (no autouse), so they cannot affect existing tests. If pytest reports a fixture-name clash, rename within this phase, not the existing tests.
- **mypy strict + numpy:** kalman/lqr store matrices typed `object` in the source (per the strong-typing convention for numpy fields). Keep the source's typing; if a faithful copy trips strict mypy, add the minimal annotation only.
- **New kalman/lqr tests:** verify the exact factory/method names (`from_config`, `initial_state`, `predict`, `update`, `compute_control`) and the state attribute name (`.x`) against the source; adjust the tests to match, never the modules.
- **Do NOT migrate `process.py`** -- the orchestration that sequences gates -> match -> ema -> kalman -> deadband -> rate -> arbiter -> lqr is the payload app shell, built in Phase 5d.
- **Legacy untouched:** `src/pact/controller/*` stays; the duplicate in `flight.payload.{tracking,gimbal}` is the go-forward.

## Self-review (against the spec)

- **Spec coverage (Section 7 tracking + gimbal):** estimators (filter/kalman) + blob tracker in `tracking/`; the FSM, the control law (lqr), and safety in `gimbal/`. The tracking-controller-as-source / arbiter-as-resolver tiering is realized through the existing pure modules; the full pipeline sequencing is the app shell (5d).
- **Placeholder scan:** no TBD/TODO in prose; new code (conftest, kalman/lqr tests) given in full with verify-against-source notes; module migrations point at exact sources with explicit import rewrites.
- **Type/name consistency:** `EmaFilterState`/`ema_update`, `KalmanFilter`/`KalmanState`/`predict`/`update`, `compute_iou`/`match_blobs`, `ArbiterState`/`GimbalArbiter`, `LqrController`/`compute_control`, and the four safety gates are used identically across modules, `__init__` re-exports, conftest, and tests.
