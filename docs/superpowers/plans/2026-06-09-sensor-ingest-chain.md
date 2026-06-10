# Sensor Ingest Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Sentinel-2 separated-band fiction with the real sensor contract: the HAL
returns a raw (H, W) uint16 2x2-mosaic plane, and demosaic + per-pixel calibration +
normalization + physically-grounded quality gates run as pure functions in
`flight/payload/preprocess`, exercised end-to-end by a raw-mosaic SIL scene.

**Architecture:** Per spec `docs/superpowers/specs/2026-06-09-pact-flight-final-state-design.md`
Section 3. `ImagingSensor.acquire_frame()` returns a new `MosaicFrame` (not a bus message --
frames never touch the bus). The payload pipeline becomes: bad-pixel repair -> dark/flat on the
mosaic plane -> 2x2 CFA separation into 4 half-resolution band planes -> [0,1] normalization ->
band selection by the new BLUE/GREEN/RED/NIR vocabulary -> quality gates (smear from slew rate x
exposure / IFOV) -> detector. Calibration artifacts are checksummed `.npy` files loaded at
startup; identity calibration is SIL-only. `RealSensor` becomes a real PySpin driver (tested via
a fake PySpin module); `sim.scene` renders radiometrically-plausible mosaic frames.

**Tech Stack:** Python 3.12+, numpy, PySpin (lazy, flight-only), pytest, mypy --strict, ruff,
import-linter. Run gates from the repo root with `uv run <tool> packages`.

**Conventions that bind every task** (from `.claude/rules/`): 100-char lines; ASCII only; full
docstrings (summary/inputs/outputs/notes + module header; tests get one-line docstrings); numpy
dtype/shape comments at declaration sites; `Result[T, E]` for library code (never raise);
`@dataclass(frozen=True, slots=True)` for data structs; enum string values mirror member names;
module docstrings cite REQ IDs. The plan's code blocks abbreviate some docstrings for space --
the executor writes them in full per the rules.

**Sensor geometry locked by this plan:** sensor 512x512 mosaic, 12-bit, 2x2 tile row-major
`(BLUE, GREEN, RED, NIR)` at cells (0,0), (0,1), (1,0), (1,1); band planes are 256x256, matching
the existing model input size. IFOV 0.04 deg/px (the old `PIXEL_TO_DEG` value, now an optics
constant in config).

**Out of scope (later phases):** ROI crop re-enable (pointing phase -- it is tracking-driven);
boresight-relative pointing math; the real `.onnx` artifact; storage of mask products; encoder
stamping of frames (gimbal phase).

---

### Task 1: Band enum, MosaicFrame type, new FaultCodes

**Files:**
- Modify: `packages/flight/src/flight/libs/types/enums.py`
- Create: `packages/flight/src/flight/libs/types/frames.py`
- Modify: `packages/flight/src/flight/libs/types/__init__.py` (add exports)
- Test: `packages/flight/tests/test_enums.py` (extend)
- Test: `packages/flight/tests/test_frames.py` (new)

- [x] **Step 1: Write the failing tests**

Append to `packages/flight/tests/test_enums.py` (follow its existing style):

```python
def test_band_values_mirror_names() -> None:
    """Band enum string values must mirror member names."""
    for member in Band:
        assert member.value == member.name


def test_new_fault_codes_exist() -> None:
    """Ingest-chain fault codes are defined with name-mirroring values."""
    assert FaultCode.CALIBRATION_INVALID.value == "CALIBRATION_INVALID"
    assert FaultCode.FRAME_MALFORMED.value == "FRAME_MALFORMED"
```

Create `packages/flight/tests/test_frames.py`:

```python
"""Tests for the MosaicFrame raw-frame value type."""

import numpy as np

from flight.libs.types import MosaicFrame


def test_mosaic_frame_holds_uint16_plane() -> None:
    """MosaicFrame carries the raw mosaic plane and capture metadata."""
    mosaic = np.zeros((4, 4), dtype=np.uint16)  # np.ndarray[uint16, (H, W)]
    frame = MosaicFrame(
        timestamp_utc="2026-06-09T00:00:00.000Z",
        frame_id=1,
        mosaic=mosaic,
        exposure_us=1000.0,
        gain_db=0.0,
    )
    assert frame.frame_id == 1
    assert np.asarray(frame.mosaic).dtype == np.uint16
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/flight/tests/test_enums.py packages/flight/tests/test_frames.py -v`
Expected: FAIL (`ImportError`/`AttributeError`: `Band`, `MosaicFrame`, new FaultCodes missing).

- [x] **Step 3: Implement**

In `enums.py`, add to `FaultCode` (after `PROCESS_DIED`):

```python
    CALIBRATION_INVALID = "CALIBRATION_INVALID"
    FRAME_MALFORMED = "FRAME_MALFORMED"
```

Add a new enum (module docstring updated accordingly):

```python
class Band(enum.Enum):
    """Physical 2x2 mosaic-filter band names.

    Passbands approximate Sentinel-2: BLUE ~490 nm (B2), GREEN ~560 nm (B3),
    RED ~665 nm (B4), NIR ~842 nm (B8) -- chosen so Sentinel-2-derived training
    data remains a valid domain (spec Section 2).
    """

    BLUE = "BLUE"
    GREEN = "GREEN"
    RED = "RED"
    NIR = "NIR"
```

Create `frames.py`:

```python
"""Raw-frame value types exchanged between the imaging HAL and the payload app.

MosaicFrame is NOT a bus message: frames are passed by direct call from the injected
sensor driver to the payload app (co-location invariant; large artifacts never go on
the bus). The mosaic plane is the un-demosaicked CFA image.

Satisfies: REQ-AIML-IMAG-001.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MosaicFrame:
    """One raw frame from the imaging sensor: CFA mosaic plane + capture metadata.

    Attributes:
        timestamp_utc: ISO 8601 capture time, millisecond precision.
        frame_id: uint32 monotonic frame counter (driver-assigned).
        mosaic: np.ndarray[uint16, (H, W)] raw 2x2-CFA mosaic plane (no processing).
        exposure_us: Exposure time in microseconds.
        gain_db: Analogue gain in dB.
    """

    timestamp_utc: str
    frame_id: int
    mosaic: object  # np.ndarray[uint16, (H, W)]
    exposure_us: float
    gain_db: float
```

Export `Band`, `MosaicFrame`, and the new FaultCodes implicitly via `flight/libs/types/__init__.py`
(add `Band` and `MosaicFrame` to its imports + `__all__`, matching its existing pattern).

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/flight/tests/test_enums.py packages/flight/tests/test_frames.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add packages/flight/src/flight/libs/types packages/flight/tests/test_enums.py packages/flight/tests/test_frames.py
git commit -m "feat(types): Band vocabulary, MosaicFrame value type, ingest fault codes"
```

---

### Task 2: SensorConfig + new preprocessing threshold

**Files:**
- Modify: `packages/flight/src/flight/libs/config/config.py`
- Modify: `config/default.toml`
- Modify: `packages/flight/src/flight/core/config_loader.py`
- Test: `packages/flight/tests/test_config_defaults.py` (extend per its existing pattern)
- Test: `packages/flight/tests/test_config_loader.py` (extend)

Note: `PreprocessingConfig.motion_smear_exposure_us` is NOT removed here (quality.py still uses
it until Task 7). This task is purely additive so the tree stays green.

- [x] **Step 1: Write the failing tests**

Extend `test_config_loader.py` with (match existing test style):

```python
def test_sensor_section_loads() -> None:
    """[sensor] TOML section maps into SensorConfig."""
    result = load_config("config/default.toml")
    assert isinstance(result, Ok)
    sensor = result.value.sensor
    assert sensor.width_px == 512
    assert sensor.height_px == 512
    assert sensor.bit_depth == 12
    assert sensor.mosaic_layout == ("BLUE", "GREEN", "RED", "NIR")
    assert sensor.ifov_deg_per_px == 0.04
    assert sensor.calibration_dir == ""
```

Extend `test_config_defaults.py` to cover the `[sensor]` section and
`preprocessing.max_motion_smear_px`, following exactly the file's existing
defaults-vs-TOML comparison pattern.

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/flight/tests/test_config_loader.py packages/flight/tests/test_config_defaults.py -v`
Expected: FAIL (`PactConfig` has no `sensor`).

- [x] **Step 3: Implement**

In `config.py` add (before `PactConfig`):

```python
@dataclass(frozen=True)
class SensorConfig:
    """Configuration for the imaging sensor and its 2x2 mosaic filter optics."""

    width_px: int = 512  # mosaic plane width in pixels (must be even)
    height_px: int = 512  # mosaic plane height in pixels (must be even)
    bit_depth: int = 12  # ADC bit depth; full scale = 2**bit_depth - 1 DN
    # Row-major band name per 2x2 cell: (0,0), (0,1), (1,0), (1,1).
    mosaic_layout: tuple[str, ...] = ("BLUE", "GREEN", "RED", "NIR")
    ifov_deg_per_px: float = 0.04  # instantaneous field of view per band-plane pixel
    default_exposure_us: float = 1000.0  # exposure commanded at startup
    default_gain_db: float = 0.0  # gain commanded at startup
    calibration_dir: str = ""  # dir of dark/flat/bad-pixel artifacts; "" -> identity (SIL only)
```

Add to `PreprocessingConfig`:

```python
    max_motion_smear_px: float = 1.0  # predicted smear (slew x exposure / IFOV) above this -> flag
```

Add to `PactConfig`: `sensor: SensorConfig = field(default_factory=SensorConfig)`.

In `config/default.toml` add (and add `max_motion_smear_px = 1.0` under `[preprocessing]`):

```toml
[sensor]
width_px = 512
height_px = 512
bit_depth = 12
mosaic_layout = ["BLUE", "GREEN", "RED", "NIR"]
ifov_deg_per_px = 0.04
default_exposure_us = 1000.0
default_gain_db = 0.0
calibration_dir = ""
```

In `config_loader.py` `_build_pact_config`, add a `sensor` block following the exact `.get()`
pattern of the other sections (str/int/float/tuple coercions matching the dataclass), import
`SensorConfig`, and pass `sensor=sensor_config` into the returned `PactConfig`. Also map
`max_motion_smear_px` in the preprocessing block.

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/flight/tests/test_config_loader.py packages/flight/tests/test_config_defaults.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add packages/flight/src/flight/libs/config/config.py config/default.toml packages/flight/src/flight/core/config_loader.py packages/flight/tests/test_config_loader.py packages/flight/tests/test_config_defaults.py
git commit -m "feat(config): SensorConfig (mosaic geometry + optics) and smear threshold"
```

---

### Task 3: Demosaic / CFA separation

**Files:**
- Create: `packages/flight/src/flight/payload/preprocess/demosaic.py`
- Modify: `packages/flight/src/flight/payload/preprocess/__init__.py` (export)
- Test: `packages/flight/tests/test_preprocess_demosaic.py`

- [x] **Step 1: Write the failing tests**

```python
"""Tests for 2x2 CFA separation and interleave round-trip."""

import numpy as np

from flight.libs.types import Err, FaultCode, Ok
from flight.payload.preprocess import interleave_bands, separate_bands


def test_separate_bands_extracts_cells() -> None:
    """Each band plane is the strided sample of its row-major 2x2 cell."""
    mosaic = np.arange(16, dtype=np.float32).reshape(4, 4)  # np.ndarray[float32, (4, 4)]
    result = separate_bands(mosaic)
    assert isinstance(result, Ok)
    planes = result.value  # np.ndarray[float32, (4, 2, 2)]
    assert planes.shape == (4, 2, 2)
    np.testing.assert_array_equal(planes[0], mosaic[0::2, 0::2])  # cell (0,0)
    np.testing.assert_array_equal(planes[1], mosaic[0::2, 1::2])  # cell (0,1)
    np.testing.assert_array_equal(planes[2], mosaic[1::2, 0::2])  # cell (1,0)
    np.testing.assert_array_equal(planes[3], mosaic[1::2, 1::2])  # cell (1,1)


def test_separate_bands_rejects_odd_or_non_2d() -> None:
    """Odd dimensions or wrong rank return Err(FRAME_MALFORMED)."""
    odd = np.zeros((5, 4), dtype=np.float32)
    result = separate_bands(odd)
    assert isinstance(result, Err)
    assert result.error == FaultCode.FRAME_MALFORMED
    assert isinstance(separate_bands(np.zeros((4,), dtype=np.float32)), Err)


def test_interleave_is_inverse_of_separate() -> None:
    """interleave_bands(separate_bands(m)) reproduces the mosaic."""
    rng = np.random.default_rng(0)
    mosaic = rng.uniform(0.0, 4095.0, size=(8, 8)).astype(np.float32)
    planes = separate_bands(mosaic)
    assert isinstance(planes, Ok)
    rebuilt = interleave_bands(planes.value)
    assert isinstance(rebuilt, Ok)
    np.testing.assert_array_equal(rebuilt.value, mosaic)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/flight/tests/test_preprocess_demosaic.py -v`
Expected: FAIL (`ImportError: separate_bands`).

- [x] **Step 3: Implement `demosaic.py`**

```python
"""2x2 CFA separation: raw mosaic plane <-> registered band planes.

The 2x2 tile repeats across the sensor; band plane k is the stride-2 sample of
row-major cell k. Planes are half the mosaic resolution and spatially registered to
each other (no interpolation -- plane co-registration error is half a mosaic pixel,
absorbed into the pointing budget). Band NAMES are assigned by SensorConfig.mosaic_layout
in the same row-major cell order; this module is layout-agnostic.

interleave_bands is the exact inverse, used by the sim scene renderer and round-trip tests.

Satisfies: REQ-AIML-PREP-001, REQ-AIML-IMAG-001.
"""

from __future__ import annotations

# stdlib
from typing import Final

# third-party
import numpy as np

# internal
from flight.libs.types import Err, FaultCode, Ok, Result

# Row-major (row_offset, col_offset) of each 2x2 cell; plane order follows this.
CELL_OFFSETS: Final[tuple[tuple[int, int], ...]] = ((0, 0), (0, 1), (1, 0), (1, 1))


def separate_bands(mosaic: np.ndarray) -> Result[np.ndarray, FaultCode]:
    """Split a (H, W) mosaic plane into (4, H/2, W/2) float32 band planes.

    Returns Err(FRAME_MALFORMED) if the input is not 2-D with even dimensions.
    """
    if mosaic.ndim != 2 or mosaic.shape[0] % 2 != 0 or mosaic.shape[1] % 2 != 0:
        return Err(FaultCode.FRAME_MALFORMED)
    planes = np.stack(
        [mosaic[r::2, c::2] for r, c in CELL_OFFSETS]
    ).astype(np.float32)  # np.ndarray[float32, (4, H/2, W/2)]
    return Ok(planes)


def interleave_bands(planes: np.ndarray) -> Result[np.ndarray, FaultCode]:
    """Rebuild the (H, W) mosaic from (4, h, w) band planes (inverse of separate_bands).

    Returns Err(FRAME_MALFORMED) if the input is not (4, h, w).
    """
    if planes.ndim != 3 or planes.shape[0] != 4:
        return Err(FaultCode.FRAME_MALFORMED)
    h, w = planes.shape[1], planes.shape[2]
    mosaic = np.empty((2 * h, 2 * w), dtype=planes.dtype)  # np.ndarray[float32, (H, W)]
    for k, (r, c) in enumerate(CELL_OFFSETS):
        mosaic[r::2, c::2] = planes[k]
    return Ok(mosaic)
```

Export both functions from `preprocess/__init__.py`.

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/flight/tests/test_preprocess_demosaic.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add packages/flight/src/flight/payload/preprocess packages/flight/tests/test_preprocess_demosaic.py
git commit -m "feat(preprocess): 2x2 CFA separation with interleave inverse"
```

---

### Task 4: Mosaic-plane radiometric calibration (bad-pixel + dark/flat)

**Files:**
- Modify: `packages/flight/src/flight/payload/preprocess/radiometric.py` (additive -- old
  `RadiometricCalibration`/`apply_calibration` stay until Task 7)
- Modify: `packages/flight/src/flight/payload/preprocess/__init__.py` (export)
- Test: `packages/flight/tests/test_preprocess_mosaic_calibration.py`

- [x] **Step 1: Write the failing tests**

```python
"""Tests for mosaic-plane calibration: bad-pixel repair then dark/flat correction."""

import numpy as np

from flight.libs.types import Err, FaultCode, Ok
from flight.payload.preprocess import MosaicCalibration, calibrate_mosaic, correct_bad_pixels


def _identity_cal(h: int, w: int) -> MosaicCalibration:
    return MosaicCalibration(
        dark_frame=np.zeros((h, w), dtype=np.float32),
        flat_field=np.ones((h, w), dtype=np.float32),
        bad_pixel_mask=np.zeros((h, w), dtype=bool),
    )


def test_correct_bad_pixels_uses_same_band_neighbors() -> None:
    """A bad pixel is replaced by the mean of its four +/-2 (same CFA cell) neighbors."""
    mosaic = np.zeros((8, 8), dtype=np.float32)
    mosaic[4, 4] = 1000.0  # the bad pixel
    mosaic[2, 4], mosaic[6, 4], mosaic[4, 2], mosaic[4, 6] = 10.0, 20.0, 30.0, 40.0
    mask = np.zeros((8, 8), dtype=bool)
    mask[4, 4] = True
    repaired = correct_bad_pixels(mosaic, mask)
    assert repaired[4, 4] == 25.0  # mean of the four same-band neighbors
    assert repaired[2, 4] == 10.0  # good pixels untouched


def test_calibrate_mosaic_applies_dark_and_flat() -> None:
    """corrected = (repaired - dark) / flat, elementwise on the mosaic plane."""
    mosaic = np.full((4, 4), 100.0, dtype=np.float32)
    cal = MosaicCalibration(
        dark_frame=np.full((4, 4), 20.0, dtype=np.float32),
        flat_field=np.full((4, 4), 2.0, dtype=np.float32),
        bad_pixel_mask=np.zeros((4, 4), dtype=bool),
    )
    result = calibrate_mosaic(mosaic, cal)
    assert isinstance(result, Ok)
    np.testing.assert_allclose(result.value, 40.0)


def test_calibrate_mosaic_shape_mismatch_is_frame_malformed() -> None:
    """A mosaic that does not match the calibration shape returns FRAME_MALFORMED."""
    result = calibrate_mosaic(np.zeros((6, 6), dtype=np.float32), _identity_cal(4, 4))
    assert isinstance(result, Err)
    assert result.error == FaultCode.FRAME_MALFORMED


def test_calibrate_mosaic_nonfinite_is_inference_nan() -> None:
    """A zero flat-field pixel produces Err(INFERENCE_NAN), never NaN output."""
    cal = _identity_cal(4, 4)
    bad_flat = cal.flat_field.copy()
    bad_flat[0, 0] = 0.0
    cal2 = MosaicCalibration(cal.dark_frame, bad_flat, cal.bad_pixel_mask)
    result = calibrate_mosaic(np.ones((4, 4), dtype=np.float32), cal2)
    assert isinstance(result, Err)
    assert result.error == FaultCode.INFERENCE_NAN
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/flight/tests/test_preprocess_mosaic_calibration.py -v`
Expected: FAIL (`ImportError: MosaicCalibration`).

- [x] **Step 3: Implement (append to `radiometric.py`)**

```python
@dataclass(frozen=True)
class MosaicCalibration:
    """Per-pixel calibration for the RAW mosaic plane (pre-demosaic).

    Loaded once at startup from checksummed artifacts (flight) or built as identity
    (SIL). Applied before CFA separation, where the physics lives.

    Attributes:
        dark_frame: np.ndarray[float32, (H, W)] per-pixel dark signal (DN).
        flat_field: np.ndarray[float32, (H, W)] normalized response map, ~1.0.
        bad_pixel_mask: np.ndarray[bool, (H, W)] True where the pixel is unusable.
    """

    dark_frame: np.ndarray  # (H, W) float32
    flat_field: np.ndarray  # (H, W) float32, values ~1.0
    bad_pixel_mask: np.ndarray  # (H, W) bool


def correct_bad_pixels(mosaic: np.ndarray, bad_pixel_mask: np.ndarray) -> np.ndarray:
    """Replace bad pixels with the mean of their four same-band (+/-2) neighbors.

    Offsets of +/-2 along each axis stay inside the same 2x2 CFA cell, so the
    replacement uses same-band data. Edge pixels use reflected padding. Single-pass:
    a bad neighbor contributes its raw value (acceptable for isolated defects;
    clustered defects should be excluded at characterization time).
    """
    padded = np.pad(mosaic, 2, mode="reflect")  # np.ndarray[float32, (H+4, W+4)]
    neighbors = (
        padded[:-4, 2:-2] + padded[4:, 2:-2] + padded[2:-2, :-4] + padded[2:-2, 4:]
    ) / 4.0  # np.ndarray[float32, (H, W)]
    return np.where(bad_pixel_mask, neighbors, mosaic).astype(np.float32)


def calibrate_mosaic(
    mosaic: np.ndarray,
    cal: MosaicCalibration,
) -> Result[np.ndarray, FaultCode]:
    """Bad-pixel repair then (repaired - dark) / flat on the raw mosaic plane.

    Returns:
        Ok(np.ndarray[float32, (H, W)]) calibrated DN;
        Err(FRAME_MALFORMED) on shape mismatch;
        Err(INFERENCE_NAN) if any output pixel is non-finite.
    """
    if mosaic.shape != cal.dark_frame.shape:
        return Err(FaultCode.FRAME_MALFORMED)
    repaired = correct_bad_pixels(mosaic, cal.bad_pixel_mask)
    corrected = (repaired - cal.dark_frame) / cal.flat_field  # np.ndarray[float32, (H, W)]
    if not np.isfinite(corrected).all():
        return Err(FaultCode.INFERENCE_NAN)
    return Ok(corrected)
```

Update the module docstring to describe both the legacy (C, H, W) path (marked: removed in the
ingest switchover) and the new mosaic-plane path. Export the three new names from
`preprocess/__init__.py`.

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/flight/tests/test_preprocess_mosaic_calibration.py packages/flight/tests/test_preprocess_radiometric.py -v`
Expected: PASS (old tests still green -- additive change).

- [x] **Step 5: Commit**

```bash
git add packages/flight/src/flight/payload/preprocess packages/flight/tests/test_preprocess_mosaic_calibration.py
git commit -m "feat(preprocess): mosaic-plane calibration with bad-pixel repair"
```

---

### Task 5: DN normalization

**Files:**
- Create: `packages/flight/src/flight/payload/preprocess/normalize.py`
- Modify: `packages/flight/src/flight/payload/preprocess/__init__.py` (export)
- Test: `packages/flight/tests/test_preprocess_normalize.py`

- [x] **Step 1: Write the failing tests**

```python
"""Tests for DN -> [0, 1] normalization."""

import numpy as np

from flight.payload.preprocess import normalize_dn


def test_normalize_scales_by_full_scale() -> None:
    """12-bit full scale (4095) maps to 1.0; zero maps to 0.0."""
    planes = np.array([[[0.0, 4095.0]]], dtype=np.float32)  # (1, 1, 2)
    out = normalize_dn(planes, bit_depth=12)
    np.testing.assert_allclose(out, [[[0.0, 1.0]]])
    assert out.dtype == np.float32


def test_normalize_clips_out_of_range() -> None:
    """Dark-subtraction undershoot and overshoot clip to [0, 1]."""
    planes = np.array([[[-10.0, 5000.0]]], dtype=np.float32)
    out = normalize_dn(planes, bit_depth=12)
    np.testing.assert_allclose(out, [[[0.0, 1.0]]])
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/flight/tests/test_preprocess_normalize.py -v`
Expected: FAIL (`ImportError: normalize_dn`).

- [x] **Step 3: Implement `normalize.py`**

```python
"""DN -> [0, 1] normalization for calibrated band planes.

normalized = clip(dn / (2**bit_depth - 1), 0, 1). This is the reflectance-like domain
the quality thresholds and the model input contract assume (spec Section 4: the model
manifest's input domain is exactly this function's output). Clipping bounds calibration
under/overshoot; saturation detection still works because saturated pixels land at 1.0.

Satisfies: REQ-AIML-PREP-002.
"""

from __future__ import annotations

# third-party
import numpy as np


def normalize_dn(planes: np.ndarray, bit_depth: int) -> np.ndarray:
    """Normalize calibrated DN band planes to [0, 1] float32 by ADC full scale.

    Args:
        planes: np.ndarray[float32, (C, H, W)] calibrated DN values.
        bit_depth: ADC bit depth; full scale is 2**bit_depth - 1.

    Returns:
        np.ndarray[float32, (C, H, W)] in [0, 1].
    """
    full_scale = float(2**bit_depth - 1)
    return np.clip(planes / full_scale, 0.0, 1.0).astype(np.float32)
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/flight/tests/test_preprocess_normalize.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add packages/flight/src/flight/payload/preprocess packages/flight/tests/test_preprocess_normalize.py
git commit -m "feat(preprocess): DN full-scale normalization"
```

---

### Task 6: Calibration artifact loading (checksummed) + identity builder

**Files:**
- Create: `packages/flight/src/flight/payload/calibration_io.py` (outside `preprocess/` --
  it does file I/O; `preprocess/` stays pure)
- Create: `data/calibration/README.md`
- Test: `packages/flight/tests/test_calibration_io.py`

- [x] **Step 1: Write the failing tests**

```python
"""Tests for calibration artifact loading and the identity builder."""

import hashlib
import json
from pathlib import Path

import numpy as np

from flight.libs.types import Err, FaultCode, Ok
from flight.payload.calibration_io import build_identity_calibration, load_calibration


def _write_artifacts(tmp_path: Path, h: int, w: int, corrupt: bool = False) -> None:
    arrays = {
        "dark_frame": np.zeros((h, w), dtype=np.float32),
        "flat_field": np.ones((h, w), dtype=np.float32),
        "bad_pixel_mask": np.zeros((h, w), dtype=bool),
    }
    manifest: dict[str, dict[str, str]] = {}
    for name, arr in arrays.items():
        fpath = tmp_path / f"{name}.npy"
        np.save(fpath, arr)
        digest = hashlib.sha256(fpath.read_bytes()).hexdigest()
        if corrupt and name == "flat_field":
            digest = "0" * 64
        manifest[name] = {"file": f"{name}.npy", "sha256": digest}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_load_calibration_happy_path(tmp_path: Path) -> None:
    """Valid artifacts + matching checksums load into a MosaicCalibration."""
    _write_artifacts(tmp_path, 8, 8)
    result = load_calibration(str(tmp_path), height_px=8, width_px=8)
    assert isinstance(result, Ok)
    assert result.value.dark_frame.shape == (8, 8)
    assert result.value.bad_pixel_mask.dtype == np.bool_


def test_load_calibration_checksum_mismatch(tmp_path: Path) -> None:
    """A checksum mismatch returns Err(CALIBRATION_INVALID)."""
    _write_artifacts(tmp_path, 8, 8, corrupt=True)
    result = load_calibration(str(tmp_path), height_px=8, width_px=8)
    assert isinstance(result, Err)
    assert result.error == FaultCode.CALIBRATION_INVALID


def test_load_calibration_shape_mismatch(tmp_path: Path) -> None:
    """Artifacts whose shape disagrees with the sensor config are rejected."""
    _write_artifacts(tmp_path, 8, 8)
    result = load_calibration(str(tmp_path), height_px=16, width_px=16)
    assert isinstance(result, Err)
    assert result.error == FaultCode.CALIBRATION_INVALID


def test_load_calibration_missing_dir() -> None:
    """A nonexistent directory returns Err(CALIBRATION_INVALID)."""
    result = load_calibration("does/not/exist", height_px=8, width_px=8)
    assert isinstance(result, Err)
    assert result.error == FaultCode.CALIBRATION_INVALID


def test_identity_calibration_shape() -> None:
    """Identity calibration: zero dark, unit flat, no bad pixels."""
    cal = build_identity_calibration(height_px=8, width_px=8)
    assert float(cal.dark_frame.sum()) == 0.0
    assert float(cal.flat_field.mean()) == 1.0
    assert not cal.bad_pixel_mask.any()
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/flight/tests/test_calibration_io.py -v`
Expected: FAIL (module missing).

- [x] **Step 3: Implement `calibration_io.py`**

```python
"""Startup-time loading of mosaic calibration artifacts (checksummed .npy files).

Lives outside preprocess/ because it performs file I/O; the preprocess package stays
pure. manifest.json maps each artifact name (dark_frame, flat_field, bad_pixel_mask)
to {"file": <name.npy>, "sha256": <hex digest>}. Any missing file, checksum mismatch,
wrong shape, or wrong dtype yields Err(CALIBRATION_INVALID) -- the composition root
treats that as an unrecoverable startup failure. build_identity_calibration is the
SIL/dev fallback (selected by SensorConfig.calibration_dir == "").

Satisfies: REQ-AIML-PREP-002.
"""

from __future__ import annotations

# stdlib
import hashlib
import json
from pathlib import Path

# third-party
import numpy as np

# internal
from flight.libs.types import Err, FaultCode, Ok, Result
from flight.payload.preprocess import MosaicCalibration

_ARTIFACT_NAMES = ("dark_frame", "flat_field", "bad_pixel_mask")


def build_identity_calibration(height_px: int, width_px: int) -> MosaicCalibration:
    """Zero dark frame, unit flat field, all-good bad-pixel mask (SIL/dev only)."""
    shape = (height_px, width_px)
    return MosaicCalibration(
        dark_frame=np.zeros(shape, dtype=np.float32),  # np.ndarray[float32, (H, W)]
        flat_field=np.ones(shape, dtype=np.float32),  # np.ndarray[float32, (H, W)]
        bad_pixel_mask=np.zeros(shape, dtype=bool),  # np.ndarray[bool, (H, W)]
    )


def load_calibration(
    calibration_dir: str,
    height_px: int,
    width_px: int,
) -> Result[MosaicCalibration, FaultCode]:
    """Load and verify dark/flat/bad-pixel artifacts from calibration_dir.

    Verifies the sha256 of each .npy against manifest.json, then shape against the
    sensor geometry. Returns Err(CALIBRATION_INVALID) on any integrity failure.
    """
    base = Path(calibration_dir)
    manifest_path = base / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return Err(FaultCode.CALIBRATION_INVALID)

    arrays: dict[str, np.ndarray] = {}
    for name in _ARTIFACT_NAMES:
        entry = manifest.get(name)
        if not isinstance(entry, dict) or "file" not in entry or "sha256" not in entry:
            return Err(FaultCode.CALIBRATION_INVALID)
        fpath = base / str(entry["file"])
        try:
            blob = fpath.read_bytes()
        except OSError:
            return Err(FaultCode.CALIBRATION_INVALID)
        if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
            return Err(FaultCode.CALIBRATION_INVALID)
        arrays[name] = np.load(fpath)

    if any(arrays[n].shape != (height_px, width_px) for n in _ARTIFACT_NAMES):
        return Err(FaultCode.CALIBRATION_INVALID)
    return Ok(
        MosaicCalibration(
            dark_frame=arrays["dark_frame"].astype(np.float32),
            flat_field=arrays["flat_field"].astype(np.float32),
            bad_pixel_mask=arrays["bad_pixel_mask"].astype(bool),
        )
    )
```

Create `data/calibration/README.md` documenting the manifest format, that artifacts come from
sensor characterization (HIL phase), that nothing binary is committed, and that an empty
`SensorConfig.calibration_dir` selects the identity calibration (SIL/dev only).

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/flight/tests/test_calibration_io.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add packages/flight/src/flight/payload/calibration_io.py data/calibration/README.md packages/flight/tests/test_calibration_io.py
git commit -m "feat(payload): checksummed calibration artifact loader + identity builder"
```

---

### Task 7: The contract switchover (HAL -> mosaic; pipeline rewire)

This is the one coordinated change: the sensor Protocol, both sensor drivers, band selection,
quality gates, the payload app pipeline, composition, main, and the sim/SIL callers all move to
the mosaic contract in a single commit. Everything used here was built (and tested) in Tasks
1-6; this task is wiring plus the reworked `band_select`/`quality`. Work through the files in
the order given; run the full gates only at the end (intermediate states do not type-check).

**Files:**
- Modify: `packages/flight/src/flight/hal/interfaces/sensor.py`
- Modify: `packages/flight/src/flight/hal/drivers_sim/sensor.py`
- Modify: `packages/flight/src/flight/hal/drivers_real/sensor.py` (signature only; full PySpin in Task 9)
- Modify: `packages/flight/src/flight/payload/preprocess/band_select.py` (rewrite)
- Modify: `packages/flight/src/flight/payload/preprocess/quality.py` (rework smear model)
- Modify: `packages/flight/src/flight/payload/preprocess/radiometric.py` (delete legacy
  `RadiometricCalibration` + `apply_calibration`)
- Modify: `packages/flight/src/flight/payload/preprocess/__init__.py`
- Modify: `packages/flight/src/flight/payload/app.py` (pipeline + slew tracking; delete
  `build_identity_calibration` here -- superseded by `calibration_io`)
- Modify: `packages/flight/src/flight/libs/config/config.py` + `config/default.toml` +
  `packages/flight/src/flight/core/config_loader.py` (remove `motion_smear_exposure_us`)
- Modify: `packages/flight/src/flight/core/composition.py` (calib parameter)
- Modify: `packages/flight/src/flight/core/main.py` (load calibration or fail startup)
- Modify: `packages/sim/src/sim/scene/plume.py` (zeroed mosaic frames; real scene in Task 10)
- Modify: `packages/sim/src/sim/sil/runner.py` (type updates)
- Delete: `packages/flight/tests/test_preprocess_radiometric.py` (superseded by Task 4 tests)
- Test (rework): `packages/flight/tests/test_sim_sensor.py`, `test_hal_interfaces.py`,
  `test_preprocess_band_select.py`, `test_preprocess_quality.py`, `test_payload_app.py`,
  `test_composition.py`; `packages/sim/tests/test_scene.py`, `test_sil_closed_loop.py`
  (check `conftest.py` and `test_real_drivers.py` for incidental references and fix)

- [x] **Step 1: HAL Protocol** -- `interfaces/sensor.py`: `acquire_frame` returns
`Result[MosaicFrame, FaultCode]`; import `MosaicFrame` from `flight.libs.types` (drop the
`flight.libs.messages` import). Docstring: drivers ACQUIRE ONLY -- no demosaic, no calibration,
no normalization inside any driver (ADR: raw-mosaic ingest contract). Other methods unchanged.

- [x] **Step 2: SimSensor** -- `drivers_sim/sensor.py`: replays `list[MosaicFrame]`; only the
type annotations and docstrings change (replay/stall logic identical).

- [x] **Step 3: RealSensor (stub)** -- `drivers_real/sensor.py`: same lazy-PySpin `__init__`;
`acquire_frame` returns `Err(FaultCode.CAMERA_STALL)` typed as `Result[MosaicFrame, FaultCode]`.

- [x] **Step 4: band_select rewrite** (full file body below; module docstring rewritten to the
BLUE/GREEN/RED/NIR vocabulary with the Sentinel-2 correspondence note; `BAND_INDICES` deleted):

```python
def select_bands(
    planes: np.ndarray,  # (4, H, W) float32, in mosaic_layout cell order
    layout: tuple[str, ...],
    band_names: tuple[str, ...],
) -> Result[np.ndarray, FaultCode]:
    """Reorder demosaicked band planes from layout order into band_names order.

    Args:
        planes: Band planes in SensorConfig.mosaic_layout (row-major cell) order.
        layout: The band name of each plane, e.g. ("BLUE", "GREEN", "RED", "NIR").
        band_names: Requested output order (InferenceConfig.input_bands).

    Returns:
        Ok(np.ndarray[float32, (len(band_names), H, W)]);
        Err(FRAME_MALFORMED) if a requested name is absent from layout or the plane
        count disagrees with layout.
    """
    if planes.ndim != 3 or planes.shape[0] != len(layout):
        return Err(FaultCode.FRAME_MALFORMED)
    try:
        indices = [layout.index(name) for name in band_names]
    except ValueError:
        return Err(FaultCode.FRAME_MALFORMED)
    return Ok(planes[indices, :, :])  # np.ndarray[float32, (len(band_names), H, W)]
```

- [x] **Step 5: quality rework** -- new signature and smear model; saturation/cloud/sunglint
heuristics keep their current logic but the band-order comments now read
`[BLUE, GREEN, RED, NIR]` (red index 2, NIR index 3 -- unchanged numerically):

```python
def compute_quality_flags(
    bands: object,  # np.ndarray[float32, (C, H, W)], order [BLUE, GREEN, RED, NIR]
    exposure_us: float,
    slew_rate_deg_per_s: float,
    ifov_deg_per_px: float,
    utc_timestamp: str,
    cfg: PreprocessingConfig,
) -> frozenset[FrameUsabilityTag]:
```

MOTION_SMEAR becomes physical (replaces the exposure-only placeholder):

```python
    # --- MOTION_SMEAR: predicted smear length in band-plane pixels ---
    smear_px = slew_rate_deg_per_s * (exposure_us * 1e-6) / ifov_deg_per_px
    if smear_px > cfg.max_motion_smear_px:
        flags.add(FrameUsabilityTag.MOTION_SMEAR)
```

Remove `motion_smear_exposure_us` from `PreprocessingConfig`, `config/default.toml`, and the
loader mapping (and from `test_config_defaults.py` if listed there explicitly).

- [x] **Step 6: delete legacy radiometric path** -- remove `RadiometricCalibration` and
`apply_calibration` from `radiometric.py`; clean the module docstring; update
`preprocess/__init__.py` exports (final export set: `CELL_OFFSETS`, `MosaicCalibration`,
`backproject_pixel`, `calibrate_mosaic`, `compute_quality_flags`, `correct_bad_pixels`,
`crop_to_roi`, `interleave_bands`, `normalize_dn`, `select_bands`, `separate_bands`).

- [x] **Step 7: app.py rework** -- `PayloadApp` field changes: `calib: MosaicCalibration`,
new `sensor_cfg: SensorConfig`; delete the local `build_identity_calibration`. `from_config`
gains a `calib: MosaicCalibration` argument and validates geometry at startup (raising
`ValueError` -- composition-root startup is the one place raising is correct):

```python
        if cfg.sensor.width_px % 2 or cfg.sensor.height_px % 2:
            raise ValueError("sensor mosaic dimensions must be even")
        if (cfg.inference.input_height_px, cfg.inference.input_width_px) != (
            cfg.sensor.height_px // 2,
            cfg.sensor.width_px // 2,
        ):
            raise ValueError("inference input size must equal sensor size / 2")
        if sorted(cfg.sensor.mosaic_layout) != sorted(b.value for b in Band):
            raise ValueError("mosaic_layout must name each Band exactly once")
        if any(b not in cfg.sensor.mosaic_layout for b in cfg.inference.input_bands):
            raise ValueError("input_bands must be a subset of mosaic_layout")
```

`process_frame(self, raw: MosaicFrame, state, now, slew_rate_deg_per_s: float = 0.0)` pipeline
(replaces the calibrate/select block; fault handling per stage mirrors the existing
`_publish_fault` + `_fault_outcome` pattern):

```python
        mosaic = np.asarray(raw.mosaic, dtype=np.float32)  # np.ndarray[float32, (H, W)]

        calibrated = calibrate_mosaic(mosaic, self.calib)
        if isinstance(calibrated, Err):
            self._publish_fault(calibrated.error, f"calibration failed frame_id={raw.frame_id}")
            return state, self._fault_outcome(raw.frame_id, calibrated.error, state)

        planes = separate_bands(calibrated.value)
        if isinstance(planes, Err):
            self._publish_fault(planes.error, f"demosaic failed frame_id={raw.frame_id}")
            return state, self._fault_outcome(raw.frame_id, planes.error, state)

        normalized = normalize_dn(planes.value, self.sensor_cfg.bit_depth)
        selected = select_bands(
            normalized, self.sensor_cfg.mosaic_layout, self.inference_cfg.input_bands
        )
        if isinstance(selected, Err):
            self._publish_fault(selected.error, f"band select failed frame_id={raw.frame_id}")
            return state, self._fault_outcome(raw.frame_id, selected.error, state)

        quality_flags = compute_quality_flags(
            selected.value,
            raw.exposure_us,
            slew_rate_deg_per_s,
            self.sensor_cfg.ifov_deg_per_px,
            raw.timestamp_utc,
            self.preprocessing_cfg,
        )
        processed = ProcessedFrameMsg(
            msg_type=MessageType.PROCESSED_FRAME,
            timestamp_utc=raw.timestamp_utc,
            frame_id=raw.frame_id,
            tensor=selected.value,  # np.ndarray[float32, (4, H/2, W/2)]
            quality_flags=quality_flags,
            crop_origin_px=(0, 0),
            scale_factor=1.0,
        )
```

`run()` computes the slew rate from consecutive gimbal encoder reads (0.0 on the first frame or
when a read fails -- the smear gate degrades gracefully):

```python
        prev_pos: GimbalPosition | None = None
        prev_pos_now = 0.0
        ...
                acq = self.sensor.acquire_frame()
                if isinstance(acq, Ok):
                    slew_rate = 0.0
                    pos_res = self.gimbal.read_position()
                    if isinstance(pos_res, Ok):
                        if prev_pos is not None and now > prev_pos_now:
                            d_az = pos_res.value.az_deg - prev_pos.az_deg
                            d_el = pos_res.value.el_deg - prev_pos.el_deg
                            slew_rate = math.hypot(d_az, d_el) / (now - prev_pos_now)
                        prev_pos = pos_res.value
                        prev_pos_now = now
                    state, _outcome = self.process_frame(acq.value, state, now, slew_rate)
```

(`import math`; import `GimbalPosition` from `flight.hal.interfaces`; drop the now-unused
`RawFrameMsg`/`ProcessedFrameMsg` imports as applicable; module header updated.)

- [x] **Step 8: composition + main** -- `composition.py`: `build_apps(config, bus, clock,
drivers, monitored, calib)` with `calib: MosaicCalibration` (import from
`flight.payload.preprocess`), passed to `PayloadApp.from_config`. `main.py`
`build_flight_system(config, bus, clock, calib)`; in `main()` after config load:

```python
    if config.sensor.calibration_dir:
        cal_result = load_calibration(
            config.sensor.calibration_dir, config.sensor.height_px, config.sensor.width_px
        )
        if not isinstance(cal_result, Ok):
            raise SystemExit(f"calibration load failed: {cal_result.error}")
        calib = cal_result.value
    else:
        calib = build_identity_calibration(config.sensor.height_px, config.sensor.width_px)
```

- [x] **Step 9: sim callers** -- `plume.py`: `build_frames` returns zeroed mosaic
`MosaicFrame`s (`FRAME_SIZE = 512`; `np.zeros((512, 512), dtype=np.uint16)`; drop
`RawFrameMsg`/`MessageType` imports; the 256x256 `plume_detector` mask is already the band-plane
size -- unchanged). `runner.py`: `frames: list[MosaicFrame]` (import from `flight.libs.types`);
`build_sil_system` passes `calib=build_identity_calibration(config.sensor.height_px,
config.sensor.width_px)` into `build_apps` (import from `flight.payload.calibration_io`).

- [x] **Step 10: rework the affected tests** -- update fixtures/builders that construct
`RawFrameMsg` to build `MosaicFrame(timestamp_utc=..., frame_id=..., mosaic=np.zeros((512, 512),
dtype=np.uint16), exposure_us=1000.0, gain_db=0.0)` (smaller geometries are fine where the test
also shrinks the config); update `select_bands`/`compute_quality_flags` call sites to the new
signatures; add a quality test for the physical smear model:

```python
def test_motion_smear_from_slew_and_exposure() -> None:
    """smear_px = slew * exposure / IFOV; above max_motion_smear_px raises the flag."""
    bands = np.zeros((4, 8, 8), dtype=np.float32)
    cfg = PreprocessingConfig()  # max_motion_smear_px = 1.0
    # 2 deg/s * 0.05 s / 0.04 deg/px = 2.5 px > 1.0 -> flagged
    flags = compute_quality_flags(bands, 50_000.0, 2.0, 0.04, "2026-06-09T00:00:00.000Z", cfg)
    assert FrameUsabilityTag.MOTION_SMEAR in flags
    # 0 deg/s -> no smear
    flags = compute_quality_flags(bands, 50_000.0, 0.0, 0.04, "2026-06-09T00:00:00.000Z", cfg)
    assert FrameUsabilityTag.MOTION_SMEAR not in flags
```

and a payload-app test that a full-pipeline frame produces a (4, H/2, W/2) tensor:

```python
def test_process_frame_demosaics_to_half_resolution() -> None:
    """A 512x512 mosaic yields a (4, 256, 256) tensor for the detector."""
    # build PayloadApp via from_config with default PactConfig, SimSensor/SimGimbal,
    # ScriptedDetector, ManualClock, identity calibration (follow the existing
    # test_payload_app.py fixture pattern), then:
    frame = MosaicFrame(
        timestamp_utc="2026-06-09T00:00:00.000Z",
        frame_id=1,
        mosaic=np.zeros((512, 512), dtype=np.uint16),
        exposure_us=1000.0,
        gain_db=0.0,
    )
    state, outcome = app.process_frame(frame, app.controller.initial_state(), now=1.0)
    assert outcome.fault is None
```

Delete `test_preprocess_radiometric.py`. Update `test_composition.py` for the `calib` parameter.

- [x] **Step 11: Run the full gates**

Run: `uv run pytest packages` then `uv run ruff check packages` then
`uv run ruff format --check packages` then `uv run mypy packages` then `uv run lint-imports`
Expected: all green. Fix fallout before committing (grep for remaining `RawFrameMsg`
constructions outside `flight.libs.messages` -- there must be none).

- [x] **Step 12: Commit**

```bash
git add -A packages config
git commit -m "feat(ingest)!: raw-mosaic sensor contract end-to-end (demosaic in preprocess)"
```

---

### Task 8: Remove RawFrameMsg from the message contract

**Files:**
- Modify: `packages/flight/src/flight/libs/messages/messages.py` (delete `RawFrameMsg`;
  update `ProcessedFrameMsg` docstring to "(4, H, W) float32, bands per
  InferenceConfig.input_bands (BLUE/GREEN/RED/NIR), H/W = sensor size / 2")
- Modify: `packages/flight/src/flight/libs/messages/__init__.py` (drop export)
- Modify: `packages/flight/src/flight/libs/types/enums.py` (delete `MessageType.RAW_FRAME`)
- Test: `packages/flight/tests/test_messages.py`, `packages/flight/tests/test_enums.py`

- [x] **Step 1: Write the failing test** -- in `test_messages.py`:

```python
def test_raw_frame_msg_removed() -> None:
    """Frames never ride the bus: RawFrameMsg and RAW_FRAME no longer exist."""
    import flight.libs.messages as messages

    assert not hasattr(messages, "RawFrameMsg")
    assert not hasattr(MessageType, "RAW_FRAME")
```

- [x] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/flight/tests/test_messages.py -v` -- Expected: FAIL.

- [x] **Step 3: Implement** -- delete the class, the enum member, the exports, and any
remaining `RawFrameMsg`/`RAW_FRAME` references in `test_messages.py`/`test_enums.py`
(`uv run python -c "..."` grep equivalent: `rg "RawFrameMsg|RAW_FRAME" packages` must return
only this plan's history). 

- [x] **Step 4: Run the full gates**

Run: `uv run pytest packages` and `uv run mypy packages` -- Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add -A packages
git commit -m "refactor(messages)!: remove RawFrameMsg -- frames never ride the bus"
```

---

### Task 9: RealSensor -- full PySpin acquisition + control plane

**Files:**
- Modify: `packages/flight/src/flight/hal/drivers_real/sensor.py` (full implementation)
- Modify: `packages/flight/src/flight/core/main.py` (`RealSensor(clock=clock)` + startup
  exposure/gain from `config.sensor`)
- Test: `packages/flight/tests/test_real_sensor_pyspin.py` (new, fake-SDK)
- Test: `packages/flight/tests/test_real_drivers.py` (keep the no-SDK ImportError test green)

- [ ] **Step 1: Write the failing tests** -- inject a fake `PySpin` module so the lazy import
inside `__init__` resolves to it:

```python
"""RealSensor behavior tests against a fake PySpin module (no SDK in CI)."""

import sys
import types

import numpy as np
import pytest

from flight.libs.time import ManualClock
from flight.libs.types import Err, FaultCode, Ok


class _FakeImage:
    def __init__(self, incomplete: bool = False) -> None:
        self._incomplete = incomplete

    def IsIncomplete(self) -> bool:  # noqa: N802 - PySpin API casing
        return self._incomplete

    def GetNDArray(self) -> np.ndarray:  # noqa: N802
        return np.full((4, 4), 100, dtype=np.uint16)

    def Release(self) -> None:  # noqa: N802
        pass


class _FakeFloatNode:
    def __init__(self, value: float) -> None:
        self._value = value

    def SetValue(self, value: float) -> None:  # noqa: N802
        self._value = value

    def GetValue(self) -> float:  # noqa: N802
        return self._value


class _FakeCamera:
    def __init__(self) -> None:
        self.ExposureTime = _FakeFloatNode(1000.0)
        self.Gain = _FakeFloatNode(0.0)
        self.next_image: _FakeImage | Exception = _FakeImage()

    def Init(self) -> None:  # noqa: N802
        pass

    def BeginAcquisition(self) -> None:  # noqa: N802
        pass

    def EndAcquisition(self) -> None:  # noqa: N802
        pass

    def GetNextImage(self, timeout_ms: int) -> _FakeImage:  # noqa: N802
        if isinstance(self.next_image, Exception):
            raise self.next_image
        return self.next_image


def _install_fake_pyspin(monkeypatch: pytest.MonkeyPatch, camera: _FakeCamera) -> None:
    fake = types.ModuleType("PySpin")

    class SpinnakerException(Exception):
        pass

    class _CamList:
        def GetSize(self) -> int:  # noqa: N802
            return 1

        def GetByIndex(self, index: int) -> _FakeCamera:  # noqa: N802
            return camera

        def GetBySerial(self, serial: str) -> _FakeCamera:  # noqa: N802
            return camera

    class _System:
        @staticmethod
        def GetInstance() -> "_System":  # noqa: N802
            return _System()

        def GetCameras(self) -> _CamList:  # noqa: N802
            return _CamList()

    fake.SpinnakerException = SpinnakerException  # type: ignore[attr-defined]
    fake.System = _System  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PySpin", fake)


def test_acquire_frame_returns_mosaic(monkeypatch: pytest.MonkeyPatch) -> None:
    """A complete image converts to a uint16 MosaicFrame with metadata."""
    from flight.hal.drivers_real import RealSensor

    camera = _FakeCamera()
    _install_fake_pyspin(monkeypatch, camera)
    sensor = RealSensor(clock=ManualClock())
    result = sensor.acquire_frame()
    assert isinstance(result, Ok)
    assert np.asarray(result.value.mosaic).dtype == np.uint16
    assert result.value.frame_id == 1
    assert result.value.exposure_us == 1000.0


def test_incomplete_image_is_camera_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    """An incomplete transfer returns Err(CAMERA_STALL)."""
    from flight.hal.drivers_real import RealSensor

    camera = _FakeCamera()
    camera.next_image = _FakeImage(incomplete=True)
    _install_fake_pyspin(monkeypatch, camera)
    sensor = RealSensor(clock=ManualClock())
    result = sensor.acquire_frame()
    assert isinstance(result, Err)
    assert result.error == FaultCode.CAMERA_STALL


def test_sdk_timeout_is_camera_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    """A SpinnakerException during GetNextImage returns Err(CAMERA_STALL)."""
    from flight.hal.drivers_real import RealSensor

    camera = _FakeCamera()
    _install_fake_pyspin(monkeypatch, camera)
    sensor = RealSensor(clock=ManualClock())
    camera.next_image = sys.modules["PySpin"].SpinnakerException("timeout")  # type: ignore[attr-defined]
    result = sensor.acquire_frame()
    assert isinstance(result, Err)
    assert result.error == FaultCode.CAMERA_STALL


def test_set_exposure_writes_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """set_exposure_us writes the camera ExposureTime node."""
    from flight.hal.drivers_real import RealSensor

    camera = _FakeCamera()
    _install_fake_pyspin(monkeypatch, camera)
    sensor = RealSensor(clock=ManualClock())
    assert isinstance(sensor.set_exposure_us(2500.0), Ok)
    assert camera.ExposureTime.GetValue() == 2500.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest packages/flight/tests/test_real_sensor_pyspin.py -v`
Expected: FAIL (`RealSensor.__init__` does not accept `clock`; behavior missing).

- [ ] **Step 3: Implement RealSensor**

```python
"""Real FLIR Blackfly S imaging-sensor driver (reference camera, spec Section 2).

PySpin imports lazily in __init__: importing the module never needs the SDK, only
constructing RealSensor does. The driver ACQUIRES ONLY -- raw uint16 mosaic out, no
image processing (ingest contract ADR). acquire_frame is serialized with a lock so
the capture loop and the control plane (exposure/gain) can run from different threads.

Satisfies: REQ-AIML-IMAG-001.
"""

from __future__ import annotations

# stdlib
import threading

# third-party
import numpy as np

# internal
from flight.libs.time import Clock
from flight.libs.types import Err, FaultCode, MosaicFrame, Ok, Result


class RealSensor:
    """FLIR Blackfly S driver over PySpin, satisfying ImagingSensor structurally."""

    def __init__(
        self,
        clock: Clock,
        serial_number: str | None = None,
        timeout_ms: int = 1000,
    ) -> None:
        """Open the camera via PySpin and initialize it.

        Raises:
            ImportError: If PySpin (the FLIR Spinnaker SDK) is not installed.
        """
        try:
            import PySpin
        except ImportError as exc:
            raise ImportError(
                "PySpin is not installed. Install the FLIR Spinnaker SDK to use "
                "RealSensor; use SimSensor in tests and simulation."
            ) from exc
        self._pyspin = PySpin
        self._system = PySpin.System.GetInstance()
        cameras = self._system.GetCameras()
        self._cam = (
            cameras.GetBySerial(serial_number) if serial_number else cameras.GetByIndex(0)
        )
        self._cam.Init()
        self._clock = clock
        self._timeout_ms = timeout_ms
        self._frame_id = 0
        self._lock = threading.Lock()

    def acquire_frame(self) -> Result[MosaicFrame, FaultCode]:
        """Capture one raw mosaic frame; Err(CAMERA_STALL) on timeout/incomplete."""
        with self._lock:
            try:
                image = self._cam.GetNextImage(self._timeout_ms)
            except self._pyspin.SpinnakerException:
                return Err(FaultCode.CAMERA_STALL)
            if image.IsIncomplete():
                image.Release()
                return Err(FaultCode.CAMERA_STALL)
            mosaic = np.array(
                image.GetNDArray(), dtype=np.uint16, copy=True
            )  # np.ndarray[uint16, (H, W)]
            image.Release()
            self._frame_id += 1
            return Ok(
                MosaicFrame(
                    timestamp_utc=self._clock.wall_clock_iso(),
                    frame_id=self._frame_id,
                    mosaic=mosaic,
                    exposure_us=float(self._cam.ExposureTime.GetValue()),
                    gain_db=float(self._cam.Gain.GetValue()),
                )
            )

    def set_exposure_us(self, exposure: float) -> Result[None, FaultCode]:
        """Write the ExposureTime node; Err(CAMERA_STALL) on an SDK error."""
        with self._lock:
            try:
                self._cam.ExposureTime.SetValue(exposure)
            except self._pyspin.SpinnakerException:
                return Err(FaultCode.CAMERA_STALL)
            return Ok(None)

    def set_gain_db(self, gain: float) -> Result[None, FaultCode]:
        """Write the Gain node; Err(CAMERA_STALL) on an SDK error."""
        with self._lock:
            try:
                self._cam.Gain.SetValue(gain)
            except self._pyspin.SpinnakerException:
                return Err(FaultCode.CAMERA_STALL)
            return Ok(None)

    def start_acquisition(self) -> Result[None, FaultCode]:
        """BeginAcquisition; Err(CAMERA_STALL) on an SDK error."""
        with self._lock:
            try:
                self._cam.BeginAcquisition()
            except self._pyspin.SpinnakerException:
                return Err(FaultCode.CAMERA_STALL)
            return Ok(None)

    def stop_acquisition(self) -> Result[None, FaultCode]:
        """EndAcquisition; Err(CAMERA_STALL) on an SDK error."""
        with self._lock:
            try:
                self._cam.EndAcquisition()
            except self._pyspin.SpinnakerException:
                return Err(FaultCode.CAMERA_STALL)
            return Ok(None)
```

In `main.py` `build_flight_system`: `sensor = RealSensor(clock=clock)`; after constructing
drivers, command the startup tuning (`sensor.set_exposure_us(config.sensor.default_exposure_us)`
and `sensor.set_gain_db(config.sensor.default_gain_db)`, ignoring `Ok`; on `Err` raise
`SystemExit` -- camera unusable at startup is unrecoverable).

Check `test_real_drivers.py`: the existing assert-ImportError-without-SDK test must still pass
(construction args changed -- update the call to `RealSensor(clock=RealClock())` equivalent
used there).

- [ ] **Step 4: Run the full gates**

Run: `uv run pytest packages` and `uv run mypy packages` -- Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/flight/src/flight/hal/drivers_real/sensor.py packages/flight/src/flight/core/main.py packages/flight/tests/test_real_sensor_pyspin.py packages/flight/tests/test_real_drivers.py
git commit -m "feat(hal): real PySpin acquisition + control plane for RealSensor"
```

---

### Task 10: Raw-mosaic scene rendering for SIL

**Files:**
- Modify: `packages/sim/src/sim/scene/plume.py`
- Test: `packages/sim/tests/test_scene.py` (extend)
- Test: `packages/sim/tests/test_sil_closed_loop.py` (verify still green on rendered frames)

- [ ] **Step 1: Write the failing tests** -- extend `test_scene.py`:

```python
def test_build_frames_renders_uint16_mosaic() -> None:
    """Rendered frames are 512x512 uint16 mosaics within 12-bit range."""
    frames = build_frames(num_frames=3, seed=7)
    assert len(frames) == 3
    mosaic = np.asarray(frames[0].mosaic)
    assert mosaic.shape == (512, 512)
    assert mosaic.dtype == np.uint16
    assert int(mosaic.max()) <= 4095


def test_build_frames_deterministic_for_seed() -> None:
    """The same seed renders identical frames (SIL determinism)."""
    a = np.asarray(build_frames(num_frames=1, seed=3)[0].mosaic)
    b = np.asarray(build_frames(num_frames=1, seed=3)[0].mosaic)
    np.testing.assert_array_equal(a, b)


def test_plume_brightens_nir_at_center() -> None:
    """The NIR plane is brighter inside the plume region than the background."""
    frames = build_frames(num_frames=1, seed=0)
    planes = separate_bands(np.asarray(frames[0].mosaic, dtype=np.float32))
    assert isinstance(planes, Ok)
    nir = planes.value[3]  # layout (BLUE, GREEN, RED, NIR) -> NIR is plane 3
    assert float(nir[110:140, 110:140].mean()) > float(nir[:40, :40].mean())
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest packages/sim/tests/test_scene.py -v`
Expected: FAIL (`build_frames` has no `seed`; zeroed mosaics fail the brightness test).

- [ ] **Step 3: Implement** -- rewrite `build_frames` in `plume.py` (module docstring updated:
the scene now renders signal through the full ingest path; `plume_detector` unchanged):

```python
FRAME_SIZE = 512  # mosaic plane size; band planes are 256x256
_BIT_DEPTH = 12
_FULL_SCALE = float(2**_BIT_DEPTH - 1)
# Background and plume amplitudes as fractions of full scale, per band plane in
# row-major cell order (BLUE, GREEN, RED, NIR). Smoke reflects strongest in NIR.
_BACKGROUND = (0.15, 0.15, 0.15, 0.18)
_PLUME_AMPLITUDE = (0.05, 0.08, 0.12, 0.25)
_PLUME_CENTER = (125.0, 125.0)  # band-plane px, inside the scripted detector mask
_PLUME_SIGMA = 12.0  # band-plane px
_NOISE_SIGMA_DN = 2.0


def build_frames(num_frames: int, seed: int = 0) -> list[MosaicFrame]:
    """Render num_frames raw mosaic frames: background + Gaussian plume + noise.

    Per band plane: dn = (background + amplitude * gaussian) * full_scale + noise,
    quantized to 12-bit uint16, then interleaved into the 2x2 CFA mosaic (the exact
    inverse of the flight demosaic). Deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    half = FRAME_SIZE // 2
    yy, xx = np.mgrid[0:half, 0:half]  # np.ndarray[int, (256, 256)] each
    gauss = np.exp(
        -(((yy - _PLUME_CENTER[0]) ** 2 + (xx - _PLUME_CENTER[1]) ** 2)
          / (2.0 * _PLUME_SIGMA**2))
    ).astype(np.float32)  # np.ndarray[float32, (256, 256)]

    frames: list[MosaicFrame] = []
    for frame_id in range(1, num_frames + 1):
        planes = np.stack(
            [
                (_BACKGROUND[k] + _PLUME_AMPLITUDE[k] * gauss) * _FULL_SCALE
                for k in range(4)
            ]
        ).astype(np.float32)  # np.ndarray[float32, (4, 256, 256)]
        planes = planes + rng.normal(0.0, _NOISE_SIGMA_DN, size=planes.shape)
        mosaic_result = interleave_bands(planes.astype(np.float32))
        assert isinstance(mosaic_result, Ok)  # geometry is fixed; cannot fail
        mosaic = np.clip(mosaic_result.value, 0.0, _FULL_SCALE).astype(
            np.uint16
        )  # np.ndarray[uint16, (512, 512)]
        frames.append(
            MosaicFrame(
                timestamp_utc="2026-06-01T00:00:00.000Z",
                frame_id=frame_id,
                mosaic=mosaic,
                exposure_us=1000.0,
                gain_db=0.0,
            )
        )
    return frames
```

(Imports: `MosaicFrame`, `Ok` from `flight.libs.types`; `interleave_bands` from
`flight.payload.preprocess`.) Update SIL tests' `build_frames(...)` calls if they pass
positional args.

- [ ] **Step 4: Run the full gates**

Run: `uv run pytest packages` -- Expected: PASS, including both SIL closed-loop tests now
running real signal through calibrate -> demosaic -> normalize -> select -> quality.

- [ ] **Step 5: Commit**

```bash
git add packages/sim
git commit -m "feat(sim): radiometrically-plausible raw-mosaic plume scene"
```

---

### Task 11: ADR + context docs + final verification

**Files:**
- Create: `docs/adr/NNNN-raw-mosaic-sensor-ingest.md` (next number after the existing ADRs --
  list `docs/adr/` first)
- Modify: `packages/flight/src/flight/payload/CONTEXT.md`, `packages/flight/src/flight/hal/CONTEXT.md`,
  `packages/flight/src/flight/libs/CONTEXT.md`, `packages/sim/src/sim/CONTEXT.md`

- [ ] **Step 1: Write the ADR** -- context (sensor-domain mismatch, baseline Section 4.4),
decision (raw-mosaic HAL contract; demosaic + calibration + normalization in preprocess as pure
functions; BLUE/GREEN/RED/NIR vocabulary with Sentinel-2 correspondence; checksummed calibration
artifacts; frames never on the bus; drivers acquire only), consequences (HAL/driver/SIL all
exercise one ingest path; model input domain defined by `normalize_dn`; band planes are
half-resolution; calibration is a startup gate). Reference spec Section 3 and the 2026-06-06
baseline.

- [ ] **Step 2: Update the four CONTEXT.md files** -- payload: the new pipeline order +
calibration injection + slew-rate smear input; hal: the acquire-only contract + MosaicFrame +
fake-PySpin test pattern; libs: Band/MosaicFrame/new FaultCodes + RawFrameMsg removal; sim: the
rendered mosaic scene + identity calibration in SIL.

- [ ] **Step 3: Run the full gates one final time**

Run: `uv run pytest packages; uv run ruff check packages; uv run ruff format --check packages; uv run mypy packages; uv run lint-imports`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add docs/adr packages
git commit -m "docs: ADR + subsystem context for the raw-mosaic ingest contract"
```

---

## Self-review notes (already applied)

- Spec Section 3 coverage: raw-mosaic HAL (T1/T7), demosaic-in-preprocess (T3/T7), bad-pixel +
  dark/flat on the raw plane (T4), normalization (T5), physical smear quality gate (T7),
  BLUE/GREEN/RED/NIR vocabulary (T1/T7), checksummed `data/calibration/` artifacts (T6),
  identity-calibration-is-SIL-only (T6/T7), PySpin acquisition + control plane (T9), raw-mosaic
  SIL scene (T10), frames-never-on-bus (T7/T8). ROI crop re-enable is explicitly deferred to the
  pointing phase (header).
- Type consistency: `MosaicFrame` (5 fields, no gimbal angles -- slew rate comes from
  `GimbalActuator.read_position()` in the app); `MosaicCalibration` named distinctly from the
  legacy `RadiometricCalibration` it replaces; `select_bands(planes, layout, band_names)` and
  `compute_quality_flags(bands, exposure_us, slew_rate_deg_per_s, ifov_deg_per_px,
  utc_timestamp, cfg)` used consistently in T7 code and tests.
- Known intentional coupling: Task 7 is one large coordinated commit because the HAL return type
  and the app pipeline cannot change independently; all nontrivial logic it wires was unit-tested
  in Tasks 1-6.
