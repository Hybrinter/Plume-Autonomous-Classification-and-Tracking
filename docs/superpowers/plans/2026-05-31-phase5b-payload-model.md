# Phase 5b -- Payload Model (DetectorBackend) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the payload's swappable detection layer in `flight.payload.model`: a pure `extract_blobs` shared by both backends, a `DetectorBackend` protocol, a deterministic `ScriptedDetector` (SIL/tests), and an `OnnxDetector` that lazy-imports `onnxruntime` over a frozen artifact -- keeping the flight image torch-free and `mypy --strict`/`ruff` clean.

**Architecture:** `flight/payload/model/blobs.py` holds `extract_blobs(prob_mask, confidence_gate, min_blob_area_px) -> tuple[BlobMeta, ...]` (connected-component analysis, pure numpy/scipy, lifted verbatim from the original InferenceEngine). `flight/payload/model/detector.py` holds the `@runtime_checkable` `DetectorBackend` protocol (`detect(ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]`), `ScriptedDetector` (runs `extract_blobs` over a fixed mask), and `OnnxDetector` (lazy `onnxruntime` import in `__init__`, like the camera SDK pattern; runs the session, sigmoid, threshold, then `extract_blobs`). The torch `InferenceEngine` and all training code remain in `src/pact` (they migrate to `tools/` in a later, off-critical-path step); flight never imports torch.

**Tech Stack:** Python 3.14, numpy, scipy.ndimage, typing.Protocol, optional onnxruntime (lazy), pytest, mypy --strict, ruff, import-linter.

---

## Context for the implementer

- Reference source (READ for the blob logic; it is currently INLINED in `InferenceEngine.run`): `src/pact/model/inference.py` lines ~120-146. The connected-component logic (threshold -> `scipy.ndimage.label` -> per-component bbox/centroid/area/mean-confidence -> `BlobMeta`) is pure numpy/scipy and is reproduced verbatim as `extract_blobs` below. Do NOT migrate the torch parts of `InferenceEngine`.
- `onnxruntime` is intentionally NOT added as a dependency (avoids py3.14 wheel/lock risk); `OnnxDetector` lazy-imports it and raises a helpful `ImportError` if absent -- identical to how `RealSensor` handles PySpin. CI has no onnxruntime, so the OnnxDetector-requires-runtime test exercises the guard.
- `scipy` IS added to the flight dependencies (needed by `extract_blobs`); scipy has py3.14 wheels and is already in the resolved environment.
- Confirm `ProcessedFrameMsg` and `InferenceResultMsg` field names against `packages/flight/src/flight/libs/messages/messages.py` before finalizing (expected: `ProcessedFrameMsg(msg_type, timestamp_utc, frame_id, tensor, quality_flags, crop_origin_px, scale_factor)`; `InferenceResultMsg(msg_type, timestamp_utc, frame_id, mask, blobs, model_version, inference_ms, mode_flags)`). Adjust constructors to the real fields; never change a message.
- MUST pass `uv run mypy packages` (strict) and `uv run ruff check packages`. Do NOT modify `src/pact/`. Stage only named files. Commit locally; no push. ASCII only. Tests annotated `-> None`. No `.importlinter` change needed (payload -> libs allowed).

## File structure (created/modified in this phase)

```
packages/flight/pyproject.toml                                # MODIFY: add scipy dependency
uv.lock                                                       # MODIFY: re-locked after scipy add
packages/flight/src/flight/payload/model/__init__.py          # re-export
packages/flight/src/flight/payload/model/blobs.py             # extract_blobs (pure)
packages/flight/src/flight/payload/model/detector.py          # DetectorBackend, ScriptedDetector, OnnxDetector
packages/flight/tests/test_extract_blobs.py                   # NEW
packages/flight/tests/test_scripted_detector.py               # NEW
packages/flight/tests/test_onnx_detector.py                   # NEW
```

---

## Task 1: Add scipy + the pure `extract_blobs`

**Files:** `packages/flight/pyproject.toml`, `uv.lock`, `model/blobs.py`, `test_extract_blobs.py`

- [ ] **Step 1: Add scipy to flight deps**

In `packages/flight/pyproject.toml`, change the `dependencies` list to:
```toml
dependencies = [
    "numpy>=1.24",
    "scipy>=1.11",
    "structlog>=23.0",
]
```

- [ ] **Step 2: Re-lock**

Run: `uv lock`
Expected: resolves and updates `uv.lock` (scipy already present transitively; py3.14 wheels exist). Exit 0.

- [ ] **Step 3: Create `model/blobs.py`**

```python
"""Backend-agnostic blob extraction from a segmentation probability mask.

Connected-component analysis turning a (H, W) probability map into discrete
BlobMeta detections. Lifted verbatim from the original InferenceEngine so the
scripted and ONNX detector backends share identical detection geometry.
"""

import numpy as np
import scipy.ndimage

from flight.libs.messages import BlobMeta


def extract_blobs(
    prob_mask: np.ndarray,
    confidence_gate: float,
    min_blob_area_px: int,
) -> tuple[BlobMeta, ...]:
    """Extract connected-component blobs from a confidence mask.

    Args:
        prob_mask: (H, W) float32 probability map in [0, 1].
        confidence_gate: Threshold at/above which a pixel counts as positive.
        min_blob_area_px: Minimum pixel count for a blob to be reported.

    Returns:
        Blobs with blob_id and persistence_count set to 0 (assigned later by the tracker).
    """
    binary_mask = (prob_mask >= confidence_gate).astype(np.uint8)
    labeled, num_features = scipy.ndimage.label(binary_mask)

    blobs: list[BlobMeta] = []
    for label_idx in range(1, num_features + 1):
        component = labeled == label_idx
        pixel_area = int(component.sum())
        if pixel_area < min_blob_area_px:
            continue
        ys, xs = np.where(component)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        cx = float(xs.mean())
        cy = float(ys.mean())
        mean_conf = float(prob_mask[component].mean())
        blobs.append(
            BlobMeta(
                blob_id=0,
                bbox=(x_min, y_min, x_max, y_max),
                centroid_raw=(cx, cy),
                pixel_area=pixel_area,
                mean_confidence=mean_conf,
                persistence_count=0,
            )
        )
    return tuple(blobs)
```

- [ ] **Step 4: Write `test_extract_blobs.py`**

```python
"""Tests for the pure blob-extraction function."""

import numpy as np

from flight.payload.model.blobs import extract_blobs


def test_extracts_two_blobs() -> None:
    """Two separated high-confidence regions yield two blobs."""
    mask = np.zeros((20, 20), dtype=np.float32)  # np.ndarray[float32, (H, W)]
    mask[2:6, 2:6] = 1.0
    mask[12:18, 12:18] = 1.0
    blobs = extract_blobs(mask, confidence_gate=0.5, min_blob_area_px=4)
    assert len(blobs) == 2
    areas = sorted(blob.pixel_area for blob in blobs)
    assert areas == [16, 36]


def test_below_min_area_excluded() -> None:
    """A region smaller than min_blob_area_px is dropped."""
    mask = np.zeros((20, 20), dtype=np.float32)
    mask[5:7, 5:7] = 1.0  # area 4
    blobs = extract_blobs(mask, confidence_gate=0.5, min_blob_area_px=10)
    assert blobs == ()


def test_bbox_and_centroid() -> None:
    """A single square blob has the expected bbox and centroid."""
    mask = np.zeros((10, 10), dtype=np.float32)
    mask[2:5, 3:6] = 1.0  # x in [3,5], y in [2,4]
    blobs = extract_blobs(mask, confidence_gate=0.5, min_blob_area_px=1)
    assert len(blobs) == 1
    blob = blobs[0]
    assert blob.bbox == (3, 2, 5, 4)
    assert blob.centroid_raw == (4.0, 3.0)
```

- [ ] **Step 5: Verify and commit**

Run: `uv run pytest packages/flight/tests/test_extract_blobs.py -v` -> PASS. `uv run mypy packages` -> Success. `uv run ruff check packages` -> passed.
```bash
git add packages/flight/pyproject.toml uv.lock packages/flight/src/flight/payload/model/blobs.py packages/flight/tests/test_extract_blobs.py
git commit -m "feat(payload): add scipy and the pure extract_blobs function"
```

---

## Task 2: DetectorBackend + ScriptedDetector + OnnxDetector

**Files:** `model/detector.py`, `model/__init__.py`, `test_scripted_detector.py`, `test_onnx_detector.py`

- [ ] **Step 1: Create `detector.py`**

```python
"""Swappable detection backends for the payload.

DetectorBackend abstracts onboard detection: SIL/tests use the deterministic
ScriptedDetector while flight uses OnnxDetector (a frozen ONNX artifact run via
onnxruntime). onnxruntime is imported lazily in OnnxDetector.__init__, so importing
this module never requires it -- mirroring the camera SDK pattern. Both backends
share extract_blobs for identical detection geometry.
"""

import time
from typing import Protocol, runtime_checkable

import numpy as np

from flight.libs.messages import InferenceResultMsg, ProcessedFrameMsg
from flight.libs.types import Err, FaultCode, MessageType, Ok, Result
from flight.payload.model.blobs import extract_blobs


@runtime_checkable
class DetectorBackend(Protocol):
    """Onboard detector: turns a preprocessed frame into a detection result."""

    def detect(self, frame: ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]:
        """Run detection on a preprocessed frame."""
        ...


class ScriptedDetector:
    """Deterministic detector backed by a fixed probability mask (SIL/tests).

    Each detect() runs the shared extract_blobs over the configured mask, exercising
    the real detection geometry without a model.
    """

    def __init__(
        self,
        prob_mask: np.ndarray,
        confidence_gate: float = 0.55,
        min_blob_area_px: int = 15,
        model_version: str = "scripted",
    ) -> None:
        """Configure the scripted mask and detection thresholds."""
        self._prob_mask = prob_mask
        self._confidence_gate = confidence_gate
        self._min_blob_area_px = min_blob_area_px
        self._model_version = model_version

    def detect(self, frame: ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]:
        """Return a detection result built from the fixed mask for this frame."""
        blobs = extract_blobs(self._prob_mask, self._confidence_gate, self._min_blob_area_px)
        return Ok(
            InferenceResultMsg(
                msg_type=MessageType.INFERENCE_RESULT,
                timestamp_utc=frame.timestamp_utc,
                frame_id=frame.frame_id,
                mask=self._prob_mask,
                blobs=blobs,
                model_version=self._model_version,
                inference_ms=0.0,
                mode_flags=0,
            )
        )


class OnnxDetector:
    """ONNX-runtime detector over a frozen model artifact (flight).

    onnxruntime is imported lazily in __init__; importing this module does not
    require it. detect() runs the session, applies sigmoid + threshold, and reuses
    extract_blobs. Exercised only when onnxruntime and a real .onnx artifact exist.
    """

    def __init__(
        self,
        model_path: str,
        confidence_gate: float = 0.55,
        min_blob_area_px: int = 15,
        model_version: str = "unknown",
    ) -> None:
        """Open an onnxruntime session over model_path.

        Raises:
            ImportError: If onnxruntime is not installed.
        """
        try:
            import onnxruntime
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is not installed. Install it and provide a frozen .onnx "
                "artifact to use OnnxDetector; use ScriptedDetector in tests and simulation."
            ) from exc
        self._session = onnxruntime.InferenceSession(model_path)
        self._confidence_gate = confidence_gate
        self._min_blob_area_px = min_blob_area_px
        self._model_version = model_version

    def detect(self, frame: ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]:
        """Run the ONNX session on the frame tensor and extract blobs."""
        start = time.perf_counter()
        bands = np.asarray(frame.tensor, dtype=np.float32)  # np.ndarray[float32, (C, H, W)]
        model_input = bands[np.newaxis, ...]  # (1, C, H, W)
        input_name = self._session.get_inputs()[0].name
        logits = self._session.run(None, {input_name: model_input})[0]  # (1, 1, H, W)
        probs = 1.0 / (1.0 + np.exp(-logits))
        if not bool(np.isfinite(probs).all()):
            return Err(FaultCode.INFERENCE_NAN)
        prob_mask = probs[0, 0].astype(np.float32)  # (H, W)
        blobs = extract_blobs(prob_mask, self._confidence_gate, self._min_blob_area_px)
        inference_ms = (time.perf_counter() - start) * 1000.0
        return Ok(
            InferenceResultMsg(
                msg_type=MessageType.INFERENCE_RESULT,
                timestamp_utc=frame.timestamp_utc,
                frame_id=frame.frame_id,
                mask=prob_mask,
                blobs=blobs,
                model_version=self._model_version,
                inference_ms=inference_ms,
                mode_flags=0,
            )
        )
```

- [ ] **Step 2: Create `model/__init__.py`**

```python
"""Payload detection: swappable backends + shared blob extraction."""

from flight.payload.model.blobs import extract_blobs
from flight.payload.model.detector import DetectorBackend, OnnxDetector, ScriptedDetector

__all__ = ["DetectorBackend", "OnnxDetector", "ScriptedDetector", "extract_blobs"]
```

- [ ] **Step 3: Write `test_scripted_detector.py`**

```python
"""Tests for the scripted detector backend."""

import numpy as np

from flight.libs.messages import ProcessedFrameMsg
from flight.libs.types import MessageType, Ok
from flight.payload.model import DetectorBackend, ScriptedDetector


def _processed_frame() -> ProcessedFrameMsg:
    """Build a minimal ProcessedFrameMsg (tensor content is unused by ScriptedDetector)."""
    tensor = np.zeros((4, 20, 20), dtype=np.float32)  # np.ndarray[float32, (C, H, W)]
    return ProcessedFrameMsg(
        msg_type=MessageType.PROCESSED_FRAME,
        timestamp_utc="2026-05-31T00:00:00.000Z",
        frame_id=7,
        tensor=tensor,
        quality_flags=frozenset(),
        crop_origin_px=(0, 0),
        scale_factor=1.0,
    )


def test_scripted_detector_returns_blobs() -> None:
    """ScriptedDetector returns Ok(InferenceResultMsg) with blobs from its mask."""
    mask = np.zeros((20, 20), dtype=np.float32)
    mask[2:8, 2:8] = 1.0
    detector = ScriptedDetector(mask, confidence_gate=0.5, min_blob_area_px=4)
    result = detector.detect(_processed_frame())
    assert isinstance(result, Ok)
    assert result.value.frame_id == 7
    assert len(result.value.blobs) == 1
    assert result.value.model_version == "scripted"


def test_scripted_detector_satisfies_protocol() -> None:
    """ScriptedDetector conforms to DetectorBackend (typed + runtime)."""
    detector: DetectorBackend = ScriptedDetector(np.zeros((4, 4), dtype=np.float32))
    assert isinstance(detector, DetectorBackend)
```

Note: confirm `ProcessedFrameMsg`'s real field names (esp. `crop_origin_px`) from the source and adjust the constructor if they differ.

- [ ] **Step 4: Write `test_onnx_detector.py`**

```python
"""Tests for the ONNX detector backend's optional-dependency guard."""

import importlib.util

import pytest

from flight.payload.model import OnnxDetector


@pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is not None,
    reason="onnxruntime is installed; the absent-runtime guard cannot be exercised",
)
def test_onnx_detector_requires_onnxruntime_when_absent() -> None:
    """Constructing OnnxDetector without onnxruntime raises a helpful ImportError."""
    with pytest.raises(ImportError):
        OnnxDetector("model.onnx")
```

- [ ] **Step 5: Verify and commit**

Run: `uv run pytest packages/flight/tests/test_scripted_detector.py packages/flight/tests/test_onnx_detector.py -v` -> PASS. `uv run mypy packages` -> Success. `uv run ruff check packages` -> passed.
```bash
git add packages/flight/src/flight/payload/model/detector.py packages/flight/src/flight/payload/model/__init__.py packages/flight/tests/test_scripted_detector.py packages/flight/tests/test_onnx_detector.py
git commit -m "feat(payload): add DetectorBackend, ScriptedDetector, OnnxDetector"
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
Expected: all pass; `lint-imports` 7 contracts kept; pytest includes the new blob/detector tests.

- [ ] **Step 2: If `ruff format --check packages` flags new files**, run `uv run ruff format packages`, re-check, commit:
```bash
git add packages
git commit -m "style: ruff-format new model files"
```
(Skip if nothing needed reformatting.)

---

## Risks & notes

- **mypy strict + untyped third-party:** `scipy.ndimage.label` and the lazily-imported `onnxruntime` are untyped; with `ignore_missing_imports` they resolve to `Any`, which strict mypy permits. Do not add `# type: ignore` unless a specific error appears; if one does, scope it narrowly.
- **No torch in flight:** do NOT import torch anywhere under `packages/`. The torch `InferenceEngine`, `architecture.py`, and all training code stay in `src/pact` and migrate to `tools/` in a later off-critical-path step.
- **onnxruntime is optional and undeclared** (lazy import) to avoid any py3.14 wheel/lock failure; revisit declaring it as a flight extra once a real exported `.onnx` artifact exists and py3.14 wheels are confirmed.
- **Field names:** verify `ProcessedFrameMsg`/`InferenceResultMsg` fields against the migrated messages before finalizing constructors.

## Self-review (against the spec)

- **Spec coverage (Section 7 model):** frozen-artifact-via-runtime model (OnnxDetector), swappable detector backend (ScriptedDetector | OnnxDetector), the retired-InferenceEngine-exception realized (no torch in flight; model is an artifact behind a runtime). `extract_blobs` shared geometry.
- **Placeholder scan:** no TBD/TODO in prose; all new code given in full; the one reference (the inlined blob logic) is reproduced verbatim with a source pointer.
- **Type/name consistency:** `DetectorBackend`/`ScriptedDetector`/`OnnxDetector`/`extract_blobs` used identically across modules, `__init__` re-export, and tests; `detect()` signature matches between protocol, implementations, and test usage.
