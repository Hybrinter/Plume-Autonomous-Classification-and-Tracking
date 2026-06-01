# Phase 5a -- Payload Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the pure preprocessing functions into `flight.payload.preprocess`, faithfully (import-path rewrites only), with adapted + new tests, `mypy --strict` and `ruff` clean, gates green.

**Architecture:** `flight/payload/preprocess/` holds the four pure modules (`band_select`, `radiometric`, `quality`, `crop`) migrated from `src/pact/preprocessing/`. Only imports change (`pact.types.*` -> `flight.libs.*`); all logic, signatures, defaults, and array conventions are preserved. These are the payload pipeline's preprocess stage; the loop that calls them (acquire -> preprocess -> infer) is wired in a later payload sub-phase.

**Tech Stack:** Python 3.14, numpy, frozen dataclasses, pytest, mypy --strict, ruff, import-linter.

---

## Context for the implementer

- Migration sources (READ them; reproduce faithfully, change ONLY imports):
  - `src/pact/preprocessing/band_select.py` -- `BAND_INDICES: Final[dict[str,int]]`, `select_bands(raw: np.ndarray, band_names: tuple[str,...]) -> np.ndarray`. No `pact.*` imports.
  - `src/pact/preprocessing/radiometric.py` -- `RadiometricCalibration` (frozen dataclass: `dark_frame`, `flat_field`), `apply_calibration(raw: np.ndarray, cal: RadiometricCalibration) -> Result[np.ndarray, FaultCode]`. Imports `FaultCode, Ok, Err, Result` from `pact.types.enums`.
  - `src/pact/preprocessing/quality.py` -- `compute_quality_flags(bands, exposure_us: float, utc_timestamp: str, cfg: PreprocessingConfig) -> frozenset[FrameUsabilityTag]`. Imports `PreprocessingConfig` from `pact.types.config`, `FrameUsabilityTag` from `pact.types.enums`.
  - `src/pact/preprocessing/crop.py` -- `crop_to_roi(bands, center_px, output_size) -> tuple[np.ndarray, tuple[int,int]]`, `backproject_pixel(px, crop_origin, scale_factor) -> tuple[int,int]`. No `pact.*` imports.
- Import rewrites:
  - `from pact.types.enums import FaultCode, Ok, Err, Result` -> `from flight.libs.types import Err, FaultCode, Ok, Result`
  - `from pact.types.config import PreprocessingConfig` -> `from flight.libs.config import PreprocessingConfig`
  - `from pact.types.enums import FrameUsabilityTag` -> `from flight.libs.types import FrameUsabilityTag`
- `flight.payload` package dir exists (empty `__init__.py` from Phase 1); `preprocess/` is a new subpackage.
- MUST pass `uv run mypy packages` (strict) and `uv run ruff check packages`. Do NOT modify `src/pact/`. Stage only named files. Commit locally; no push. ASCII only. New test functions annotated `-> None`. No `.importlinter` change is needed (payload -> libs is allowed by the flight-layers contract).

## File structure (created in this phase)

```
packages/flight/src/flight/payload/preprocess/__init__.py     # re-export public API
packages/flight/src/flight/payload/preprocess/band_select.py  # from src/pact/preprocessing/band_select.py
packages/flight/src/flight/payload/preprocess/radiometric.py  # from src/pact/preprocessing/radiometric.py
packages/flight/src/flight/payload/preprocess/quality.py      # from src/pact/preprocessing/quality.py
packages/flight/src/flight/payload/preprocess/crop.py         # from src/pact/preprocessing/crop.py
packages/flight/tests/test_preprocess_band_select.py          # adapt from tests/unit/preprocessing/test_band_select.py
packages/flight/tests/test_preprocess_radiometric.py          # adapt from tests/unit/preprocessing/test_radiometric.py
packages/flight/tests/test_preprocess_quality.py              # adapt from tests/unit/preprocessing/test_quality.py
packages/flight/tests/test_preprocess_crop.py                 # NEW
```

---

## Task 1: Migrate the four preprocess modules + the package __init__

**Files:** the 4 modules + `__init__.py` listed above.

- [ ] **Step 1: Copy `band_select.py` and `crop.py` verbatim** (they have no `pact.*` imports) into `flight/payload/preprocess/`. Preserve the `from __future__ import annotations`, the requirement-ID docstring header, `BAND_INDICES`, and all signatures/logic exactly.

- [ ] **Step 2: Copy `radiometric.py`**, changing only the import line to `from flight.libs.types import Err, FaultCode, Ok, Result`. Preserve `RadiometricCalibration` and `apply_calibration` exactly.

- [ ] **Step 3: Copy `quality.py`**, changing only the imports to `from flight.libs.config import PreprocessingConfig` and `from flight.libs.types import FrameUsabilityTag`. Preserve `compute_quality_flags`, the `SATURATION_PIXEL_LEVEL` constant, and all heuristics exactly.

- [ ] **Step 4: Create `preprocess/__init__.py`**

```python
"""Payload preprocessing: pure functions transforming raw bands for inference.

Stage order in the payload loop: select bands -> radiometric correction ->
quality flags -> crop. All functions are pure (no I/O, no global state).
"""

from flight.payload.preprocess.band_select import BAND_INDICES, select_bands
from flight.payload.preprocess.crop import backproject_pixel, crop_to_roi
from flight.payload.preprocess.quality import compute_quality_flags
from flight.payload.preprocess.radiometric import RadiometricCalibration, apply_calibration

__all__ = [
    "BAND_INDICES",
    "RadiometricCalibration",
    "apply_calibration",
    "backproject_pixel",
    "compute_quality_flags",
    "crop_to_roi",
    "select_bands",
]
```

- [ ] **Step 5: Verify and commit**

Run: `uv run mypy packages` -> Success. `uv run ruff check packages` -> passed.
```bash
git add packages/flight/src/flight/payload/preprocess
git commit -m "feat(payload): migrate preprocessing pure functions into flight.payload.preprocess"
```

---

## Task 2: Migrate + add the preprocessing tests

**Files:** the 4 test files listed above.

- [ ] **Step 1: Adapt the three existing tests.** Copy `tests/unit/preprocessing/test_band_select.py`, `test_radiometric.py`, `test_quality.py` into `packages/flight/tests/test_preprocess_band_select.py`, `test_preprocess_radiometric.py`, `test_preprocess_quality.py`. Rewrite imports: source functions from `flight.payload.preprocess`; `FaultCode`/`Ok`/`Err`/`FrameUsabilityTag` from `flight.libs.types`; `PreprocessingConfig` from `flight.libs.config`. Keep the local helpers and assertions identical. Ensure every test function is annotated `-> None`.

- [ ] **Step 2: Write the new `test_preprocess_crop.py`** (crop has no existing test). First READ `src/pact/preprocessing/crop.py` to confirm exact semantics (whether `center_px` is (x, y), how `output_size` maps to the crop, the origin convention, and the `backproject_pixel` scale formula); adjust the expected values below to match the source if they differ.

```python
"""Tests for the ROI crop and back-projection preprocessing functions."""

import numpy as np

from flight.payload.preprocess import backproject_pixel, crop_to_roi


def test_crop_to_roi_returns_requested_size() -> None:
    """crop_to_roi returns an array of the requested output size and float32 dtype."""
    bands = np.zeros((4, 100, 100), dtype=np.float32)  # np.ndarray[float32, (C, H, W)]
    cropped, origin = crop_to_roi(bands, center_px=(50, 50), output_size=(20, 20))
    assert cropped.shape == (4, 20, 20)
    assert cropped.dtype == np.float32
    assert isinstance(origin, tuple)
    assert len(origin) == 2


def test_backproject_pixel_round_trips_with_crop_origin() -> None:
    """backproject_pixel adds the crop origin back at unit scale."""
    bands = np.zeros((4, 100, 100), dtype=np.float32)
    _, origin = crop_to_roi(bands, center_px=(50, 50), output_size=(20, 20))
    full = backproject_pixel(px=(0, 0), crop_origin=origin, scale_factor=1.0)
    assert full == origin
```

- [ ] **Step 2: Verify and commit**

Run: `uv run pytest packages/flight/tests/test_preprocess_band_select.py packages/flight/tests/test_preprocess_radiometric.py packages/flight/tests/test_preprocess_quality.py packages/flight/tests/test_preprocess_crop.py -v` -> PASS. `uv run mypy packages` -> Success. `uv run ruff check packages` -> passed.
```bash
git add packages/flight/tests/test_preprocess_band_select.py packages/flight/tests/test_preprocess_radiometric.py packages/flight/tests/test_preprocess_quality.py packages/flight/tests/test_preprocess_crop.py
git commit -m "test(payload): migrate and add preprocessing tests"
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
Expected: all pass; `lint-imports` 7 contracts kept; pytest includes the new preprocess tests.

- [ ] **Step 2: If `ruff format --check packages` flags new files**, run `uv run ruff format packages`, re-check, commit:
```bash
git add packages
git commit -m "style: ruff-format new preprocess files"
```
(Skip if nothing needed reformatting.)

---

## Risks & notes

- **Crop semantics:** the new crop test must match the source's actual conventions (center ordering, origin, scale). Verify against `src/pact/preprocessing/crop.py` and adjust the test, never the migrated function.
- **mypy strict:** `compute_quality_flags`'s `bands` parameter is typed `object` in the source (numpy array); keep it as-is. If a faithful copy trips strict mypy, add the minimal annotation only.
- **Legacy untouched:** `src/pact/preprocessing/*` stays; the duplicate in `flight.payload.preprocess` is the go-forward and will be consumed by the payload loop in a later sub-phase.

## Self-review (against the spec)

- **Spec coverage (Section 7 payload preprocess stage):** all four pure modules migrated; the loop wiring is explicitly deferred to a later payload sub-phase.
- **Placeholder scan:** no TBD/TODO in prose; the new crop test is given in full with a verify-against-source note; migrations point at exact sources with explicit import rewrites.
- **Type/name consistency:** `select_bands`, `apply_calibration`, `RadiometricCalibration`, `compute_quality_flags`, `crop_to_roi`, `backproject_pixel`, `BAND_INDICES` are used identically across the modules, the `__init__` re-export, and the tests.
