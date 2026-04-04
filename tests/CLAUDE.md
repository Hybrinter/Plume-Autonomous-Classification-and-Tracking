# PACT Test Suite — Implementation Guide

This file is the authoritative reference for writing and running PACT tests. It supplements
§6 of `PACT_SW_ARCH.md` with implementation-level detail. When this file and the arch spec
conflict, raise the discrepancy — do not silently follow either.

---

## 1. Directory Structure

```
tests/
├── CLAUDE.md                        ← this file
├── conftest.py                      ← shared fixtures (see §3)
├── unit/
│   ├── model/
│   │   ├── test_architecture.py
│   │   ├── test_dataset.py
│   │   └── test_evaluate.py
│   ├── preprocessing/
│   │   ├── test_band_select.py
│   │   ├── test_radiometric.py
│   │   └── test_quality.py
│   ├── controller/
│   │   ├── test_arbiter.py
│   │   ├── test_tracker.py
│   │   ├── test_filter.py
│   │   └── test_safety.py
│   ├── comms/
│   │   ├── test_ccsds.py
│   │   └── test_scheduler.py
│   ├── storage/
│   │   ├── test_writer.py
│   │   └── test_manifest.py
│   └── fault/
│       ├── test_detector.py
│       └── test_watchdog.py
└── integration/
    ├── test_inference_pipeline.py
    ├── test_controller_pipeline.py
    ├── test_comms_pipeline.py
    └── e2e/
        └── test_full_pipeline_smoke.py
```

---

## 2. Test Conventions

### 2.1 Imports

Unit tests import **only**:
- The module under test (e.g., `from pact.controller.arbiter import GimbalArbiter, ArbiterState`)
- `pact.types.*` (enums, messages, config dataclasses)
- `pytest` and `pytest.mark`
- `numpy as np` where synthetic arrays are needed

Do not import other subsystem modules in unit tests. If a unit test needs a message type that
a different subsystem produces, construct it directly using the frozen dataclass constructor.

### 2.2 No real processes, files, or hardware in unit tests

- No `multiprocessing.Process` or `subprocess` in unit tests.
- No file I/O. Use `tmp_path` fixture (pytest built-in) only in storage/manifest tests where
  file I/O is the thing under test, and even then mock the filesystem if possible.
- No `FlirBlackflyCamera`. Always use `MockCamera`.
- No network sockets.

### 2.3 Parametrize all threshold and boundary tests

Every function that has a numeric threshold, gate, or limit must be tested with
`pytest.mark.parametrize` across at minimum:
- A value clearly below the threshold
- A value at the threshold (boundary — test both sides: `threshold - epsilon`, `threshold`)
- A value clearly above the threshold

```python
@pytest.mark.parametrize("confidence,expected_count", [
    (0.30, 0),   # well below gate
    (0.54, 0),   # just below gate (0.55)
    (0.55, 1),   # at gate exactly — should pass
    (0.90, 1),   # well above gate
])
def test_confidence_gate(confidence: float, expected_count: int) -> None:
    blobs = (make_blob(mean_confidence=confidence),)
    result = apply_confidence_gate(blobs, threshold=0.55)
    assert len(result) == expected_count
```

### 2.4 Test coverage requirements

Each test file must cover, at minimum:
- **Happy path** — the function returns `Ok(expected_value)` or the expected output.
- **Error/fault path** — the function returns `Err(expected_fault_code)` or raises the
  expected exception (for process entry points only).
- **Boundary values** — per §2.3 above.

### 2.5 Result type assertions

Functions returning `Result[T, E]` should be unwrapped explicitly in tests:

```python
result = apply_calibration(raw, cal)
assert isinstance(result, Ok), f"Expected Ok, got Err({result.error})"
assert result.value.shape == expected_shape
```

Do not use `result.value` without first asserting `isinstance(result, Ok)`. This ensures
the test fails with a meaningful message if the function returns an unexpected `Err`.

### 2.6 Line length

100 characters, same as source. Configure your editor accordingly.

---

## 3. Fixture Catalogue

All shared fixtures are defined in `tests/conftest.py`. Do not redefine them in individual
test files — import them via pytest's automatic fixture injection.

### `mock_camera`

```python
def mock_camera() -> MockCamera
```

A `MockCamera` instance pre-loaded with 5 synthetic `RawFrameMsg` frames. Each frame has a
4-band `(4, 256, 256)` float32 array filled with uniform random values in `[0, 1]`. No blobs
are injected by default.

**When to use:** Any test that needs to call `camera.acquire_frame()` without hardware.

**Customisation:** Construct `MockCamera(frames=[...])` directly in your test if you need
specific frame content (e.g., frames with known pixel patterns for quality flag testing).

### `sample_raw_frame_msg`

```python
def sample_raw_frame_msg() -> RawFrameMsg
```

A single `RawFrameMsg` with:
- `frame_id = 1`
- `raw_bands`: `np.zeros((4, 256, 256), dtype=np.float32)`
- `exposure_us = 10000.0`, `gain_db = 0.0`
- `gimbal_az_deg = 0.0`, `gimbal_el_deg = 0.0`
- `timestamp_utc`: ISO 8601 string fixed at `"2026-04-03T00:00:00.000Z"`

**When to use:** Tests for preprocessing functions that need a `RawFrameMsg` as input.

### `sample_processed_frame_msg`

```python
def sample_processed_frame_msg() -> ProcessedFrameMsg
```

A `ProcessedFrameMsg` with:
- `frame_id = 1`
- `tensor`: `np.zeros((4, 256, 256), dtype=np.float32)`
- `quality_flags = frozenset()` (no flags — frame is usable)
- `crop_origin_px = (0, 0)`, `scale_factor = 1.0`

**When to use:** Tests for `InferenceEngine.run()` and controller input handling.

**Note:** The zero tensor will produce near-zero model outputs. If you need a specific
segmentation output, inject a synthetic `InferenceResultMsg` directly (see §5).

### `sample_inference_result`

```python
def sample_inference_result() -> InferenceResultMsg
```

An `InferenceResultMsg` with:
- `frame_id = 1`
- `mask`: `np.zeros((256, 256), dtype=np.float32)`
- `blobs`: a single `BlobMeta` with `mean_confidence=0.85`, `pixel_area=200`,
  `persistence_count=1`, `blob_id=1`, `bbox=(100, 100, 150, 150)`,
  `centroid_raw=(125.0, 125.0)`
- `model_version = "test-v0"`, `inference_ms = 50.0`, `mode_flags = 0`

**When to use:** Controller unit tests (arbiter, safety gates, tracker) where you need
a realistic inference result with one detected blob above all thresholds.

**Customisation:** Construct `InferenceResultMsg(blobs=(...), ...)` directly to vary blob
count, confidence, area, or persistence.

### `default_config`

```python
def default_config() -> PactConfig
```

A `PactConfig` loaded from `config/default.toml` via `load_config()`. All fields match
the defaults in §10 of the arch spec.

**When to use:** Any test that needs realistic threshold values (confidence gate, deadband,
etc.) without hardcoding them.

**Warning:** Do not mutate this fixture. It is frozen (`@dataclass(frozen=True)`). Construct
a new config with `dataclasses.replace(default_config.controller, confidence_gate=0.9)` if
you need a modified value.

### `arbiter_idle_state`

```python
def arbiter_idle_state() -> ArbiterState
```

An `ArbiterState` with:
- `gimbal_state = GimbalState.IDLE`
- `tracked_blobs = ()`
- `idle_duration_s = 0.0`
- `last_command_time = 0.0`
- `current_target_id = None`

**When to use:** As the starting state for all arbiter state machine tests.

---

## 4. Run Commands

### Fast unit tests (CI default, ~seconds)

```bash
pytest tests/unit/ -v
```

### Integration tests (excludes e2e, ~30 seconds)

```bash
pytest tests/integration/ -m "not e2e" -v
```

### Everything except e2e (recommended before committing)

```bash
pytest -m "not e2e" -v
```

### e2e smoke test only (slow, requires all subsystems)

```bash
pytest tests/integration/e2e/ -v --timeout=60
```

The e2e test is marked `@pytest.mark.e2e`. It must complete in under 60 seconds per the
arch spec. `pytest-timeout` enforces this hard limit.

### Coverage report

```bash
pytest -m "not e2e" --cov=pact --cov-report=term-missing
```

There is no enforced coverage threshold yet. Once the initial implementation is complete,
set `--cov-fail-under=80` in `pyproject.toml` under `[tool.pytest.ini_options]`.

### Type checking

```bash
mypy src/pact/ --strict
```

Run mypy before opening a PR. All source files must pass `--strict`. Test files are exempt
from `--strict` but must at minimum pass default mypy.

---

## 5. InferenceResultMsg Injection Point (e2e Test)

### What it is

The e2e smoke test (`tests/integration/e2e/test_full_pipeline_smoke.py`) exercises the full
system with a randomly initialized (untrained) `PactSegmentationModel`. A random model
produces garbage segmentation outputs — the mask will not contain meaningful blobs above
the confidence threshold. This means the arbiter would never leave `GimbalState.IDLE` and
assertions 3–5 in §6.4 of the arch spec would always fail.

To make the e2e test deterministic and meaningful, synthetic `InferenceResultMsg` values are
**injected directly onto the inference result queue**, bypassing the actual model output for
the frames that need to trigger ACQUIRING and TRACKING.

### Where the injection happens

The injection point is the `inference_queue: multiprocessing.Queue[InferenceResultMsg]`
that sits between the inference process and the controller process. In the e2e test:

1. The real inference process runs and consumes `RawFrameMsg` values from the imaging queue.
2. For frames 1–3 (the frames that should trigger blob detection), the test fixture intercepts
   the real inference result and replaces it with a synthetic `InferenceResultMsg` that
   contains one blob with `mean_confidence=0.85`, `pixel_area=200`, and
   `persistence_count` incrementing 1 → 2 → 3.
3. For frames 4–10, the real inference result (garbage output, no blobs) is passed through.

The practical mechanism: a proxy queue or a thin shim process sits between the real inference
process output and the controller input. The shim replaces results for the first 3 frames
and passes the rest through.

### Why this matters

This injection point is the exact boundary between model output and controller input. It is
the interface that will matter when real model weights are integrated:
- If real weights produce correct segmentation, the injected synthetic results will be removed
  and the real results will drive the state machine.
- If real weights underperform, the injection point makes it easy to bisect: is the controller
  correct given good inputs? (Yes, proven by e2e with injection.) Is the model producing
  good inputs? (Measure separately with evaluate.py.)

### Adding new e2e assertions

Any new e2e assertion that depends on specific segmentation output must use this injection
point. Do not add e2e tests that expect the random model to produce specific mask values —
they will be flaky.

Mark all e2e tests with `@pytest.mark.e2e` so they can be excluded from fast CI runs.
