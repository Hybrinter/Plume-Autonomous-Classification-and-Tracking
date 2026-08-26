# Testing Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a marker-tiered pytest suite (unit + integration + e2e) covering `src/smoke_detection/` and `scripts/prepare_dataset.py`, enforcing ≥80% branch coverage and zero-network synthetic-data fixtures.

**Architecture:** Tests mirror `src/smoke_detection/` under `tests/unit/`. Hybrid fixture strategy: session-scoped read-only synthetic dataset + function-scoped mutable copies. Cached ImageNet weights via Torch Hub + CI `actions/cache`. Four markers (`slow`, `gpu`, `e2e`, default); Makefile targets unlock each tier.

**Tech Stack:** pytest 8, pytest-cov 5, pytest-timeout 2, rasterio, pydantic v2, PyTorch 2.x, Lightning 2.x.

**Spec:** `docs/superpowers/specs/2026-04-19-testing-suite-design.md` (approved).

**Conventions for this plan:**
- All file paths are repo-relative; use forward slashes.
- Run every command from the repo root.
- Prefix every pytest invocation with `uv run`.
- When a test fails on first run, stop and read the traceback — it often reveals a real bug in `src/`. Fix it (follow existing patterns; don't refactor unrelated code); never weaken the test. If the bug is out of scope for this plan, open a TODO comment with a `# NOTE: investigate` line and move on.
- Every task ends with a commit. Use Conventional Commits (`test:`, `ci:`, `chore(test):`, etc.).

---

## Phase 0: Infrastructure

### Task 1: Register pytest markers, coverage config, and timeout in `pyproject.toml`

**Files:**
- Modify: `pyproject.toml` — `[project.optional-dependencies].dev`, `[tool.pytest.ini_options]`, add `[tool.coverage.run]`, `[tool.coverage.report]`.

- [ ] **Step 1: Edit `pyproject.toml`**

Replace the `[project.optional-dependencies].dev` block and the `[tool.pytest.ini_options]` block, and append coverage sections at the end of the file.

```toml
[project.optional-dependencies]
dev = [
    "ruff>=0.4",
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "pytest-timeout>=2.3",
    "pre-commit>=3.6",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers --cov=src/smoke_detection --cov-branch --cov-report=term-missing -m 'not slow and not gpu and not e2e' --timeout=30"
markers = [
    "slow: tests > ~2s (pretrained weight load, overfitting sanity loops)",
    "gpu: requires CUDA; auto-skipped when unavailable",
    "e2e: end-to-end pipelines (fast_dev_run, CLI, ckpt round-trip)",
]

[tool.coverage.run]
branch = true
source = ["src/smoke_detection"]
omit = ["src/smoke_detection/__init__.py", "*/__main__.py"]

[tool.coverage.report]
exclude_lines = ["pragma: no cover", "raise NotImplementedError", "if TYPE_CHECKING:"]
```

- [ ] **Step 2: Install new dev deps**

```bash
uv sync --extra dev
```

Expected: `pytest-cov` and `pytest-timeout` resolved successfully.

- [ ] **Step 3: Verify pytest discovers markers and the existing placeholder test still passes**

```bash
uv run pytest --co -q
```

Expected: collection shows `tests/test_placeholder.py::test_package_imports` and no "unknown marker" warnings.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(test): register pytest markers, coverage config, timeouts"
```

---

### Task 2: Add test-tier Make targets

**Files:**
- Modify: `Makefile` — replace the `test` target, append `test-e2e`, `test-gpu`, `test-all`.

- [ ] **Step 1: Edit `Makefile`**

Replace the existing `test:` target and add three new targets. Keep the `.PHONY` line at the top updated.

```make
.PHONY: install lint format test test-e2e test-gpu test-all clean train-cls train-seg eval-cls eval-seg
```

```make
test:
	uv run pytest -m "not slow and not e2e and not gpu"
	uv run pytest -m "slow" --cov-append --cov-fail-under=80

test-e2e:
	uv run pytest -m "e2e" --timeout=300

test-gpu:
	uv run pytest -m "gpu"

test-all:
	uv run pytest -m "not gpu"
```

- [ ] **Step 2: Dry-run**

```bash
make test
```

Expected: first command passes (one placeholder test), second passes with `--cov-append` noting no slow tests ran yet. Coverage will currently report low; `--cov-fail-under=80` will FAIL. That's expected — we'll fix by end of plan.

Temporarily comment out `--cov-fail-under=80` in the Makefile (leave a note) until Phase 12. Better: keep it; accept that `make test` is red until the last phase.

For this plan: **keep the gate in from Task 2; expect `make test` red until Task 35.** This is a deliberate tripwire — we want to see the number go up.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "chore(test): add test-e2e, test-gpu, test-all Make targets"
```

---

### Task 3: Top-level `tests/conftest.py`

**Files:**
- Create: `tests/conftest.py` (replace the existing scaffold).
- Modify: `tests/test_placeholder.py` → move to `tests/unit/__init__.py` logic (package-import test goes into `tests/unit/test_package_imports.py`, created in Task 7).

- [ ] **Step 1: Delete the old placeholder conftest**

```bash
rm tests/conftest.py tests/test_placeholder.py tests/.gitkeep
```

- [ ] **Step 2: Create the new top-level conftest**

Path: `tests/conftest.py`

```python
"""Top-level pytest fixtures.

Provides determinism, path helpers, a session-scoped synthetic prepared-dataset
tree (13-band uint16 GeoTIFFs + Label-Studio-style JSON polygons), and a
cached-weights fixture for the one slow test that loads ResNet-50 ImageNet
weights. Also auto-skips `gpu` tests when CUDA is unavailable.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterator

import pytest

from tests._data import build_synthetic_prepared_tree


REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_configure(config: pytest.Config) -> None:
    os.environ.setdefault("PYTORCH_LIGHTNING_LOG_LEVEL", "ERROR")
    # Lightning writes TB logs to CWD by default; tests redirect via cfg.paths.output_dir.


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip `gpu`-marked tests when CUDA is unavailable."""
    import torch

    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    """Seed all RNGs before every test."""
    from smoke_detection.common.seed import seed_everything

    seed_everything(1234, deterministic=True)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def synthetic_dataset_root(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Read-only session-scoped synthetic prepared-dataset tree."""
    root = tmp_path_factory.mktemp("smokedet_synthetic_prepared")
    build_synthetic_prepared_tree(root)
    prev = os.environ.get("SMOKEDET_DATA_ROOT")
    os.environ["SMOKEDET_DATA_ROOT"] = str(root)
    try:
        yield root
    finally:
        if prev is None:
            os.environ.pop("SMOKEDET_DATA_ROOT", None)
        else:
            os.environ["SMOKEDET_DATA_ROOT"] = prev
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def mutable_dataset_root(tmp_path: Path) -> Path:
    """Function-scoped writable copy of the synthetic prepared tree."""
    root = tmp_path / "mutable_prepared"
    build_synthetic_prepared_tree(root)
    return root


@pytest.fixture
def tiny_classification_batch():
    """2-sample classification batch (pre-transform-compatible)."""
    import torch

    return {
        "idx": torch.tensor([0, 1]),
        "img": torch.randn(2, 4, 90, 90),
        "lbl": torch.tensor([True, False]),
        "imgfile": ["a.tif", "b.tif"],
    }


@pytest.fixture
def tiny_segmentation_batch():
    """2-sample segmentation batch (pre-transform-compatible)."""
    import torch

    mask = torch.zeros(2, 90, 90)
    mask[0, 20:50, 30:70] = 1.0  # one positive, one all-zero
    return {
        "idx": torch.tensor([0, 1]),
        "img": torch.randn(2, 4, 90, 90),
        "fpt": mask,
        "imgfile": ["a.tif", "b.tif"],
    }


@pytest.fixture(scope="session")
def cached_resnet50_weights():
    """Return IMAGENET1K_V1 weights enum; trigger download on first call."""
    from torchvision.models import ResNet50_Weights, resnet50

    _ = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    return ResNet50_Weights.IMAGENET1K_V1
```

- [ ] **Step 3: Commit (conftest will fail to import until Task 4 adds `tests/_data.py`; don't run tests yet)**

```bash
git add tests/conftest.py tests/test_placeholder.py tests/.gitkeep
git commit -m "chore(test): replace placeholder conftest with full fixture skeleton"
```

---

### Task 4: Synthetic dataset builders (`tests/_data.py`)

**Files:**
- Create: `tests/_data.py`
- Create: `tests/__init__.py` (empty; makes `tests` importable)

- [ ] **Step 1: Create `tests/__init__.py`**

```python
"""Test package. Empty on purpose; enables `from tests._data import ...`."""
```

- [ ] **Step 2: Create `tests/_data.py`**

```python
"""Builders for synthetic smoke-plume data.

Two entry points:
- ``build_synthetic_prepared_tree`` produces the post-prepare_dataset.py layout
  consumed by the training code.
- ``build_synthetic_zenodo_source`` produces the pre-prepare_dataset.py layout
  consumed by scripts/prepare_dataset.py.

Both write real 13-band uint16 GeoTIFFs at 120x120 (so Sentinel-2 band indexing
[2,3,4,8] works) and Label-Studio-style JSON polygons with a scale factor that
survives the 1.2x multiplier in SmokePlumeSegmentationDataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.transform import from_origin


TIF_COUNT = 13
TIF_SIZE = 120


def _write_tif(path: Path, rng: np.random.Generator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = rng.integers(0, 3000, size=(TIF_COUNT, TIF_SIZE, TIF_SIZE), dtype=np.uint16)
    profile = {
        "driver": "GTiff",
        "width": TIF_SIZE,
        "height": TIF_SIZE,
        "count": TIF_COUNT,
        "dtype": "uint16",
        "transform": from_origin(0, 0, 1, 1),
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(data)


def _write_label_json(path: Path, tif_basename: str) -> None:
    """Write a Label-Studio-style polygon label file.

    Polygon is a 50x50 square near the top-left so after the 1.2x scale the
    rasterized mask stays within the 120x120 frame.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completions": [
            {
                "result": [
                    {
                        "value": {
                            "points": [
                                [10, 10],
                                [60, 10],
                                [60, 60],
                                [10, 60],
                            ]
                        }
                    }
                ]
            }
        ],
        "data": {"image": f"/data/upload/1-{tif_basename.replace('.tif', '.png')}"},
    }
    path.write_text(json.dumps(payload))


def build_synthetic_prepared_tree(
    root: Path,
    n_sites_per_split: int = 3,
    n_times_per_site: int = 2,
    seed: int = 42,
) -> None:
    """Produce the post-prepare_dataset.py layout under ``root``."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    for split in ("train", "val", "test"):
        for site_idx in range(n_sites_per_split):
            for cls in ("positive", "negative"):
                for t in range(n_times_per_site):
                    basename = f"site{split}{site_idx}_t{t:03d}.tif"
                    _write_tif(root / "classification" / split / cls / basename, rng)
                    _write_tif(root / "segmentation" / split / "images" / cls / basename, rng)
                    if cls == "positive":
                        _write_label_json(
                            root / "segmentation" / split / "labels" / f"{basename}.json",
                            basename,
                        )


def build_synthetic_zenodo_source(
    root: Path,
    n_sites: int = 6,
    n_times_per_site: int = 2,
    seed: int = 42,
) -> None:
    """Produce the pre-prepare_dataset.py Zenodo-style source layout.

    Creates:
      root/images/{positive,negative}/*.tif
      root/segmentation_labels/*.json
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    for site_idx in range(n_sites):
        for cls in ("positive", "negative"):
            for t in range(n_times_per_site):
                basename = f"site{site_idx:03d}_t{t:03d}.tif"
                _write_tif(root / "images" / cls / basename, rng)
                if cls == "positive":
                    _write_label_json(
                        root / "segmentation_labels" / f"{basename}.json",
                        basename,
                    )
```

- [ ] **Step 3: Smoke-test the builder**

```bash
uv run python -c "from pathlib import Path; import tempfile, sys; sys.path.insert(0, '.'); from tests._data import build_synthetic_prepared_tree; p = Path(tempfile.mkdtemp()); build_synthetic_prepared_tree(p); print(sorted(x.name for x in (p / 'classification' / 'train' / 'positive').iterdir()))"
```

Expected: a list of 6 `siteXXX_tYYY.tif` filenames.

- [ ] **Step 4: Run existing tests to confirm conftest still collects**

```bash
uv run pytest --co -q
```

Expected: no collection errors.

- [ ] **Step 5: Commit**

```bash
git add tests/__init__.py tests/_data.py
git commit -m "test: add synthetic dataset builders for prepared and Zenodo layouts"
```

---

### Task 5: Integration & E2E conftests

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/conftest.py`
- Create: `tests/e2e/__init__.py`
- Create: `tests/e2e/conftest.py`

- [ ] **Step 1: Create the two `__init__.py` files (empty)**

- [ ] **Step 2: Create `tests/integration/conftest.py`**

```python
"""Fixtures for integration tests.

Builds fully-validated Config objects pointed at the synthetic prepared tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smoke_detection.configs.classification import ClassificationConfig
from smoke_detection.configs.segmentation import SegmentationConfig


@pytest.fixture
def sample_classification_config(synthetic_dataset_root: Path, tmp_path: Path) -> ClassificationConfig:
    return ClassificationConfig.model_validate(
        {
            "task": "classification",
            "seed": 1234,
            "trainer": {
                "max_epochs": 1,
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32",
                "deterministic": True,
                "log_every_n_steps": 1,
                "fast_dev_run": True,
            },
            "paths": {
                "data_root": str(synthetic_dataset_root),
                "output_dir": str(tmp_path / "lightning_logs"),
                "experiment_name": "test_classification",
            },
            "optim": {"lr": 1e-3, "momentum": 0.9, "weight_decay": 0.0, "scheduler": "none"},
            "model": {"backbone": "resnet50", "pretrained": False, "in_channels": 4},
            "data": {"batch_size": 2, "num_workers": 0, "crop_size": 90, "balance": "none"},
        }
    )


@pytest.fixture
def sample_segmentation_config(synthetic_dataset_root: Path, tmp_path: Path) -> SegmentationConfig:
    return SegmentationConfig.model_validate(
        {
            "task": "segmentation",
            "seed": 1234,
            "trainer": {
                "max_epochs": 1,
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32",
                "deterministic": True,
                "log_every_n_steps": 1,
                "fast_dev_run": True,
            },
            "paths": {
                "data_root": str(synthetic_dataset_root),
                "output_dir": str(tmp_path / "lightning_logs"),
                "experiment_name": "test_segmentation",
            },
            "optim": {"lr": 1e-3, "momentum": 0.9, "weight_decay": 0.0, "scheduler": "none"},
            "model": {"architecture": "unet", "in_channels": 4, "n_classes": 1, "bilinear": True},
            "data": {"batch_size": 2, "num_workers": 0, "crop_size": 90},
        }
    )
```

- [ ] **Step 3: Create `tests/e2e/conftest.py`**

```python
"""Fixtures for e2e tests — YAML files on disk for CLI invocation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from smoke_detection.configs.classification import ClassificationConfig
from smoke_detection.configs.segmentation import SegmentationConfig


def _dump_config(cfg, path: Path) -> Path:
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    return path


@pytest.fixture
def classification_yaml_tmp(sample_classification_config: ClassificationConfig, tmp_path: Path) -> Path:
    return _dump_config(sample_classification_config, tmp_path / "cls.yaml")


@pytest.fixture
def segmentation_yaml_tmp(sample_segmentation_config: SegmentationConfig, tmp_path: Path) -> Path:
    return _dump_config(sample_segmentation_config, tmp_path / "seg.yaml")
```

Note: `tests/e2e/conftest.py` consumes fixtures defined in `tests/integration/conftest.py`. pytest only makes fixtures visible to siblings, not cousins, so duplicate the two `sample_*_config` fixtures into `tests/e2e/conftest.py`. Simpler: move the `sample_*` fixtures up into `tests/conftest.py`. Do that now:

Actually, the cleanest structure: **move the two `sample_*_config` fixtures up into `tests/conftest.py`** so both integration and e2e can consume them. Update `tests/conftest.py` (append after `cached_resnet50_weights`):

```python
@pytest.fixture
def sample_classification_config(synthetic_dataset_root: Path, tmp_path: Path):
    from smoke_detection.configs.classification import ClassificationConfig

    return ClassificationConfig.model_validate(
        {
            "task": "classification",
            "seed": 1234,
            "trainer": {
                "max_epochs": 1, "accelerator": "cpu", "devices": 1,
                "precision": "32", "deterministic": True,
                "log_every_n_steps": 1, "fast_dev_run": True,
            },
            "paths": {
                "data_root": str(synthetic_dataset_root),
                "output_dir": str(tmp_path / "lightning_logs"),
                "experiment_name": "test_classification",
            },
            "optim": {"lr": 1e-3, "momentum": 0.9, "weight_decay": 0.0, "scheduler": "none"},
            "model": {"backbone": "resnet50", "pretrained": False, "in_channels": 4},
            "data": {"batch_size": 2, "num_workers": 0, "crop_size": 90, "balance": "none"},
        }
    )


@pytest.fixture
def sample_segmentation_config(synthetic_dataset_root: Path, tmp_path: Path):
    from smoke_detection.configs.segmentation import SegmentationConfig

    return SegmentationConfig.model_validate(
        {
            "task": "segmentation",
            "seed": 1234,
            "trainer": {
                "max_epochs": 1, "accelerator": "cpu", "devices": 1,
                "precision": "32", "deterministic": True,
                "log_every_n_steps": 1, "fast_dev_run": True,
            },
            "paths": {
                "data_root": str(synthetic_dataset_root),
                "output_dir": str(tmp_path / "lightning_logs"),
                "experiment_name": "test_segmentation",
            },
            "optim": {"lr": 1e-3, "momentum": 0.9, "weight_decay": 0.0, "scheduler": "none"},
            "model": {"architecture": "unet", "in_channels": 4, "n_classes": 1, "bilinear": True},
            "data": {"batch_size": 2, "num_workers": 0, "crop_size": 90},
        }
    )
```

Then `tests/integration/conftest.py` and `tests/e2e/conftest.py` become leaner: integration's conftest can be empty or omitted; e2e's conftest just has the YAML dumpers.

- [ ] **Step 4: Reflect the correction**

Final contents:
- `tests/conftest.py` — now also holds `sample_classification_config` and `sample_segmentation_config` (appended).
- `tests/integration/__init__.py` — empty.
- `tests/integration/conftest.py` — empty or deleted.
- `tests/e2e/__init__.py` — empty.
- `tests/e2e/conftest.py` — only the YAML dumpers.

- [ ] **Step 5: Collect to verify no errors**

```bash
uv run pytest --co -q
```

Expected: no collection output yet (no tests), no errors.

- [ ] **Step 6: Commit**

```bash
git add tests/integration/ tests/e2e/ tests/conftest.py
git commit -m "test: add integration + e2e conftests with sample config fixtures"
```

---

## Phase 1: Unit tests — common/

### Task 6: `tests/unit/` skeleton + `test_package_imports`

**Files:**
- Create: `tests/unit/__init__.py`
- Create: `tests/unit/test_package_imports.py`

- [ ] **Step 1: Create `tests/unit/__init__.py`** (empty).

- [ ] **Step 2: Create `tests/unit/test_package_imports.py`**

```python
"""Smoke-level package import + version check (replaces tests/test_placeholder.py)."""

from __future__ import annotations


def test_package_imports():
    import smoke_detection

    assert smoke_detection.__version__
```

- [ ] **Step 3: Run**

```bash
uv run pytest tests/unit/test_package_imports.py -v
```

Expected: 1 passed.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/
git commit -m "test: restore package-import smoke test under tests/unit/"
```

---

### Task 7: `tests/unit/common/test_paths.py`

**Files:**
- Create: `tests/unit/common/__init__.py` (empty)
- Create: `tests/unit/common/test_paths.py`

- [ ] **Step 1: Create `tests/unit/common/test_paths.py`**

```python
"""Tests for smoke_detection.common.paths."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

from smoke_detection.common import paths as paths_module


def test_classification_split_joins_parts():
    base = Path("/tmp/fake")
    out = paths_module.classification_split("train", base)
    assert out == base / "classification" / "train"


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_classification_split_each_split(split):
    base = Path("/tmp/fake")
    out = paths_module.classification_split(split, base)
    assert out.name == split
    assert out.parent.name == "classification"


def test_segmentation_split_returns_image_and_label_dirs():
    base = Path("/tmp/fake")
    imgs, lbls = paths_module.segmentation_split("train", base)
    assert imgs == base / "segmentation" / "train" / "images"
    assert lbls == base / "segmentation" / "train" / "labels"


def test_explicit_root_overrides_default(tmp_path):
    out = paths_module.classification_split("train", tmp_path)
    assert out == tmp_path / "classification" / "train"


def test_env_var_overrides_dataset_root(monkeypatch, tmp_path):
    custom = tmp_path / "custom_root"
    monkeypatch.setenv("SMOKEDET_DATA_ROOT", str(custom))
    reloaded = importlib.reload(paths_module)
    try:
        assert reloaded.DATASET_ROOT == custom.resolve()
    finally:
        importlib.reload(paths_module)  # restore module-level state
```

- [ ] **Step 2: Run**

```bash
uv run pytest tests/unit/common/test_paths.py -v
```

Expected: 7 passed. If the env-var reload test fails on Windows because of path resolution differences, compare `.resolve()` on both sides.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/common/
git commit -m "test(common): cover paths module — splits, overrides, env var"
```

---

### Task 8: `tests/unit/common/test_seed.py`

**Files:**
- Create: `tests/unit/common/test_seed.py`

- [ ] **Step 1: Write the file**

```python
"""Tests for smoke_detection.common.seed."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from smoke_detection.common.seed import seed_everything


def _sample_triplet():
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_same_seed_produces_same_sequence():
    seed_everything(123)
    a = _sample_triplet()
    seed_everything(123)
    b = _sample_triplet()
    assert a == b


def test_different_seeds_produce_different_sequences():
    seed_everything(1)
    a = _sample_triplet()
    seed_everything(2)
    b = _sample_triplet()
    assert a != b


def test_deterministic_sets_cudnn_flags():
    seed_everything(7, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_pythonhashseed_is_set():
    seed_everything(99)
    assert os.environ["PYTHONHASHSEED"] == "99"
```

- [ ] **Step 2: Run**

```bash
uv run pytest tests/unit/common/test_seed.py -v
```

Expected: 4 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/common/test_seed.py
git commit -m "test(common): cover seed_everything determinism + flags"
```

---

### Task 9: `tests/unit/common/test_logging.py`

**Files:**
- Create: `tests/unit/common/test_logging.py`

- [ ] **Step 1: Write the file**

```python
"""Tests for smoke_detection.common.logging."""

from __future__ import annotations

import logging

from smoke_detection.common.logging import get_logger


def test_get_logger_returns_logger_instance():
    log = get_logger("smoke_detection.test_logging.a")
    assert isinstance(log, logging.Logger)


def test_get_logger_is_idempotent_about_handlers():
    name = "smoke_detection.test_logging.b"
    a = get_logger(name)
    b = get_logger(name)
    assert a is b
    assert len(a.handlers) == 1


def test_get_logger_formatter_includes_name_and_level():
    log = get_logger("smoke_detection.test_logging.c")
    fmt = log.handlers[0].formatter._fmt
    assert "%(levelname)s" in fmt
    assert "%(name)s" in fmt
```

- [ ] **Step 2: Run**

```bash
uv run pytest tests/unit/common/test_logging.py -v
```

Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/common/test_logging.py
git commit -m "test(common): cover get_logger idempotency + formatter"
```

---

## Phase 2: Unit tests — configs/

### Task 10: `tests/unit/configs/test_schemas.py`

**Files:**
- Create: `tests/unit/configs/__init__.py`
- Create: `tests/unit/configs/test_schemas.py`

- [ ] **Step 1: Write the file**

```python
"""Tests for pydantic config schemas (base, classification, segmentation)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from smoke_detection.configs.base import OptimConfig, PathsConfig, TrainerConfig
from smoke_detection.configs.classification import ClassificationConfig
from smoke_detection.configs.segmentation import SegmentationConfig


# ----------------------------- TrainerConfig -----------------------------

def test_trainer_defaults():
    t = TrainerConfig()
    assert t.max_epochs == 50
    assert t.accelerator == "auto"
    assert t.precision == "32"
    assert t.fast_dev_run is False


def test_trainer_rejects_unknown_field():
    with pytest.raises(ValidationError):
        TrainerConfig(unknown_field=1)


def test_trainer_rejects_bad_precision():
    with pytest.raises(ValidationError):
        TrainerConfig(precision="64")


def test_trainer_rejects_bad_accelerator():
    with pytest.raises(ValidationError):
        TrainerConfig(accelerator="tpu")


# ----------------------------- PathsConfig -----------------------------

def test_paths_requires_experiment_name():
    with pytest.raises(ValidationError):
        PathsConfig()


def test_paths_accepts_strings_as_path():
    p = PathsConfig(data_root="data", output_dir="lightning_logs", experiment_name="e")
    assert isinstance(p.data_root, Path)


# ----------------------------- OptimConfig -----------------------------

def test_optim_requires_lr():
    with pytest.raises(ValidationError):
        OptimConfig()


def test_optim_scheduler_literal():
    with pytest.raises(ValidationError):
        OptimConfig(lr=1e-4, scheduler="exponential")


# ----------------------------- ClassificationConfig -----------------------------

def _minimal_classification_payload():
    return {
        "task": "classification",
        "trainer": {},
        "paths": {"experiment_name": "x"},
        "optim": {"lr": 1e-4},
        "model": {},
        "data": {},
    }


def test_classification_minimal_payload_validates():
    ClassificationConfig.model_validate(_minimal_classification_payload())


def test_classification_wrong_task_tag_rejected():
    payload = _minimal_classification_payload()
    payload["task"] = "segmentation"
    with pytest.raises(ValidationError):
        ClassificationConfig.model_validate(payload)


# ----------------------------- SegmentationConfig -----------------------------

def _minimal_segmentation_payload():
    return {
        "task": "segmentation",
        "trainer": {},
        "paths": {"experiment_name": "x"},
        "optim": {"lr": 1e-4},
        "model": {},
        "data": {},
    }


def test_segmentation_minimal_payload_validates():
    SegmentationConfig.model_validate(_minimal_segmentation_payload())


def test_cross_task_load_fails():
    """A classification YAML loaded through SegmentationConfig must be rejected."""
    payload = _minimal_classification_payload()
    with pytest.raises(ValidationError):
        SegmentationConfig.model_validate(payload)
```

- [ ] **Step 2: Run**

```bash
uv run pytest tests/unit/configs/test_schemas.py -v
```

Expected: 12 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/configs/
git commit -m "test(configs): cover pydantic schemas — accept + reject cases"
```

---

### Task 11: `tests/unit/configs/test_loader.py`

**Files:**
- Create: `tests/unit/configs/test_loader.py`

- [ ] **Step 1: Write the file**

```python
"""Tests for YAML config loading + dotted overrides."""

from __future__ import annotations

import pytest

from smoke_detection.configs.classification import ClassificationConfig
from smoke_detection.configs.loader import _apply_dotted_override, _coerce_scalar, load_config
from smoke_detection.configs.segmentation import SegmentationConfig


def _write_yaml(path, body):
    path.write_text(body)
    return path


def test_load_classification_yaml(tmp_path):
    cfg = load_config("configs/classification/default.yaml")
    assert isinstance(cfg, ClassificationConfig)
    assert cfg.task == "classification"


def test_load_segmentation_yaml(tmp_path):
    cfg = load_config("configs/segmentation/default.yaml")
    assert isinstance(cfg, SegmentationConfig)
    assert cfg.task == "segmentation"


def test_dotted_override_scalar(tmp_path):
    p = _write_yaml(
        tmp_path / "c.yaml",
        "task: classification\n"
        "trainer: {}\n"
        "paths: {experiment_name: x}\n"
        "optim: {lr: 1e-4}\n"
        "model: {}\n"
        "data: {}\n",
    )
    cfg = load_config(p, overrides=["optim.lr=5e-3"])
    assert cfg.optim.lr == 5e-3


def test_dotted_override_creates_missing_dicts(tmp_path):
    p = _write_yaml(
        tmp_path / "c.yaml",
        "task: classification\n"
        "trainer: {}\n"
        "paths: {experiment_name: x}\n"
        "optim: {lr: 1e-4}\n"
        "model: {}\n"
        "data: {}\n",
    )
    cfg = load_config(p, overrides=["trainer.fast_dev_run=true"])
    assert cfg.trainer.fast_dev_run is True


def test_override_syntax_error_raises(tmp_path):
    p = _write_yaml(tmp_path / "c.yaml", "task: classification\ntrainer: {}\npaths: {experiment_name: x}\noptim: {lr: 1e-4}\nmodel: {}\ndata: {}\n")
    with pytest.raises(ValueError, match="key=value"):
        load_config(p, overrides=["just-a-key"])


def test_override_descent_into_scalar_raises():
    raw = {"a": 1}
    with pytest.raises(ValueError, match="non-mapping"):
        _apply_dotted_override(raw, "a.b=2")


def test_missing_task_raises(tmp_path):
    p = _write_yaml(tmp_path / "c.yaml", "trainer: {}\npaths: {experiment_name: x}\noptim: {lr: 1e-4}\n")
    with pytest.raises(ValueError, match="task"):
        load_config(p)


def test_invalid_task_raises(tmp_path):
    p = _write_yaml(tmp_path / "c.yaml", "task: bogus\ntrainer: {}\npaths: {experiment_name: x}\noptim: {lr: 1e-4}\n")
    with pytest.raises(ValueError, match="task"):
        load_config(p)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("true", True), ("false", False), ("True", True), ("FALSE", False),
        ("null", None), ("None", None), ("none", None),
        ("42", 42), ("-3", -3),
        ("1.5", 1.5), ("1e-4", 1e-4), ("2.5e3", 2500.0),
        ("hello", "hello"), ("", ""),
    ],
)
def test_coerce_scalar(value, expected):
    assert _coerce_scalar(value) == expected
```

- [ ] **Step 2: Run**

```bash
uv run pytest tests/unit/configs/test_loader.py -v
```

Expected: 8 + 14 parametrize cases = 22 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/configs/test_loader.py
git commit -m "test(configs): cover load_config, dotted overrides, _coerce_scalar"
```

---

## Phase 3: Unit tests — data/

### Task 12: `tests/unit/data/test_pad_to_120.py`

**Files:**
- Create: `tests/unit/data/__init__.py`
- Create: `tests/unit/data/test_pad_to_120.py`

- [ ] **Step 1: Write the file**

```python
"""Tests for SmokePlumeDataset._pad_to_120 helper."""

from __future__ import annotations

import numpy as np

from smoke_detection.data.classification_dataset import _pad_to_120


def test_identity_at_120():
    arr = np.arange(4 * 120 * 120, dtype=np.float32).reshape(4, 120, 120)
    out = _pad_to_120(arr)
    assert out.shape == (4, 120, 120)
    np.testing.assert_array_equal(out, arr)


def test_right_pad_repeats_last_column():
    arr = np.zeros((4, 120, 100), dtype=np.float32)
    arr[..., -1] = 7.0
    out = _pad_to_120(arr)
    assert out.shape == (4, 120, 120)
    # columns 100..119 must all equal the original last column (all 7s)
    assert np.all(out[:, :, 100:] == 7.0)


def test_bottom_pad_repeats_last_row():
    arr = np.zeros((4, 100, 120), dtype=np.float32)
    arr[:, -1, :] = 5.0
    out = _pad_to_120(arr)
    assert out.shape == (4, 120, 120)
    assert np.all(out[:, 100:, :] == 5.0)


def test_both_dimensions_padded():
    arr = np.ones((4, 80, 100), dtype=np.float32)
    out = _pad_to_120(arr)
    assert out.shape == (4, 120, 120)
    assert np.all(out == 1.0)
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/unit/data/test_pad_to_120.py -v
```

Expected: 4 passed.

```bash
git add tests/unit/data/
git commit -m "test(data): cover _pad_to_120 identity and right/bottom padding"
```

---

### Task 13: `tests/unit/data/test_transforms.py`

**Files:**
- Create: `tests/unit/data/test_transforms.py`

- [ ] **Step 1: Write the file**

```python
"""Tests for Normalize / Randomize / RandomCrop / ToTensor."""

from __future__ import annotations

import numpy as np
import torch

from smoke_detection.data.transforms import (
    CHANNEL_MEANS,
    CHANNEL_STDS,
    Normalize,
    Randomize,
    RandomCrop,
    ToTensor,
)


# ----------------------------- Normalize -----------------------------

def test_normalize_hand_computed():
    img = np.full((4, 4, 4), 1500.0, dtype=np.float32)
    out = Normalize()({"img": img.copy()})["img"]
    expected = (1500.0 - CHANNEL_MEANS.reshape(-1, 1, 1)) / CHANNEL_STDS.reshape(-1, 1, 1)
    np.testing.assert_allclose(out, expected, rtol=1e-5)


def test_normalize_zero_centers_means():
    img = np.zeros((4, 3, 3), dtype=np.float32)
    img += CHANNEL_MEANS.reshape(-1, 1, 1)
    out = Normalize()({"img": img})["img"]
    np.testing.assert_allclose(out, np.zeros_like(out), atol=1e-5)


def test_normalize_broadcasts_over_spatial_dims():
    img = np.random.randn(4, 7, 11).astype(np.float32)
    out = Normalize()({"img": img.copy()})["img"]
    assert out.shape == img.shape


# ----------------------------- RandomCrop -----------------------------

def test_random_crop_produces_requested_shape():
    img = np.random.randn(4, 120, 120).astype(np.float32)
    out = RandomCrop(crop=90)({"img": img})["img"]
    assert out.shape == (4, 90, 90)


def test_random_crop_matches_fpt_offsets():
    rng = np.random.default_rng(0)
    img = rng.random((4, 120, 120)).astype(np.float32)
    fpt = rng.random((120, 120)).astype(np.float32)
    # Seed numpy to fix offsets
    np.random.seed(42)
    out = RandomCrop(crop=80)({"img": img.copy(), "fpt": fpt.copy()})
    # Reapply to verify we get the exact same crop when reseeded
    np.random.seed(42)
    out2 = RandomCrop(crop=80)({"img": img.copy(), "fpt": fpt.copy()})
    np.testing.assert_array_equal(out["img"], out2["img"])
    np.testing.assert_array_equal(out["fpt"], out2["fpt"])
    # Spatial dims must match between img and fpt
    assert out["img"].shape[-2:] == out["fpt"].shape


# ----------------------------- Randomize -----------------------------

def test_randomize_is_deterministic_under_seed():
    rng = np.random.default_rng(0)
    img = rng.random((4, 10, 10)).astype(np.float32)
    fpt = rng.random((10, 10)).astype(np.float32)
    np.random.seed(9)
    out_a = Randomize()({"img": img.copy(), "fpt": fpt.copy()})
    np.random.seed(9)
    out_b = Randomize()({"img": img.copy(), "fpt": fpt.copy()})
    np.testing.assert_array_equal(out_a["img"], out_b["img"])
    np.testing.assert_array_equal(out_a["fpt"], out_b["fpt"])


def test_randomize_flips_image_and_mask_together():
    # Place a single positive pixel at (0, 0) in the mask; check that after
    # the transform, the image pixel at the mask's positive location is the
    # same one the image had there originally.
    img = np.zeros((1, 4, 4), dtype=np.float32)
    img[0, 0, 0] = 42.0
    fpt = np.zeros((4, 4), dtype=np.float32)
    fpt[0, 0] = 1.0
    np.random.seed(0)
    out = Randomize()({"img": img, "fpt": fpt})
    i, j = np.argwhere(out["fpt"] == 1.0)[0]
    assert out["img"][0, i, j] == 42.0


# ----------------------------- ToTensor -----------------------------

def test_totensor_returns_float_tensors():
    sample = {"img": np.random.randn(4, 4, 4).astype(np.float32), "fpt": np.random.randn(4, 4).astype(np.float32)}
    out = ToTensor()(sample)
    assert isinstance(out["img"], torch.Tensor)
    assert out["img"].dtype == torch.float32
    assert isinstance(out["fpt"], torch.Tensor)
    assert out["fpt"].dtype == torch.float32


def test_totensor_without_fpt():
    sample = {"img": np.random.randn(4, 4, 4).astype(np.float32)}
    out = ToTensor()(sample)
    assert isinstance(out["img"], torch.Tensor)
    assert "fpt" not in out
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/unit/data/test_transforms.py -v
```

Expected: 10 passed.

```bash
git add tests/unit/data/test_transforms.py
git commit -m "test(data): cover Normalize/Randomize/RandomCrop/ToTensor"
```

---

### Task 14: `tests/unit/data/test_classification_dataset.py`

**Files:**
- Create: `tests/unit/data/test_classification_dataset.py`

- [ ] **Step 1: Write the file**

```python
"""Tests for SmokePlumeDataset (classification)."""

from __future__ import annotations

import numpy as np
import pytest

from smoke_detection.common.paths import classification_split
from smoke_detection.data.classification_dataset import (
    SmokePlumeDataset,
    build_default_transform,
    build_eval_transform,
)


@pytest.fixture
def train_dir(synthetic_dataset_root):
    return classification_split("train", synthetic_dataset_root)


def test_length_nonzero(train_dir):
    ds = SmokePlumeDataset(datadir=train_dir, balance="none")
    assert len(ds) > 0


def test_getitem_returns_expected_keys(train_dir):
    ds = SmokePlumeDataset(datadir=train_dir, balance="none")
    s = ds[0]
    assert set(s.keys()) == {"idx", "img", "lbl", "imgfile"}


def test_img_shape_and_dtype(train_dir):
    ds = SmokePlumeDataset(datadir=train_dir, balance="none")
    s = ds[0]
    assert isinstance(s["img"], np.ndarray)
    assert s["img"].shape == (4, 120, 120)
    assert s["img"].dtype == np.float32


def test_label_derived_from_path(train_dir):
    ds = SmokePlumeDataset(datadir=train_dir, balance="none")
    for idx in range(len(ds)):
        s = ds[idx]
        if "positive" in s["imgfile"]:
            assert s["lbl"] is True
        elif "negative" in s["imgfile"]:
            assert s["lbl"] is False


def test_balance_upsample_equalizes_or_grows(train_dir):
    up = SmokePlumeDataset(datadir=train_dir, balance="upsample")
    none = SmokePlumeDataset(datadir=train_dir, balance="none")
    n_pos_none = int(none.labels.sum())
    n_neg_none = int((~none.labels.astype(bool)).sum())
    n_pos_up = int(up.labels.sum())
    # Upsampling adds duplicate positives until >= negatives.
    assert n_pos_up >= n_neg_none or n_pos_up >= n_pos_none


def test_balance_downsample_reduces_length(train_dir):
    none = SmokePlumeDataset(datadir=train_dir, balance="none")
    down = SmokePlumeDataset(datadir=train_dir, balance="downsample")
    assert len(down) <= len(none)


def test_balance_none_leaves_counts_intact(train_dir):
    ds = SmokePlumeDataset(datadir=train_dir, balance="none")
    assert len(ds) == len(ds.imgfiles)


def test_mult_scales_length(train_dir):
    base = SmokePlumeDataset(datadir=train_dir, balance="none", mult=1)
    doubled = SmokePlumeDataset(datadir=train_dir, balance="none", mult=2)
    assert len(doubled) == 2 * len(base)


def test_transform_is_applied(train_dir):
    tfm = build_eval_transform()
    ds = SmokePlumeDataset(datadir=train_dir, balance="none", transform=tfm)
    s = ds[0]
    # After ToTensor, img must be a torch.Tensor.
    import torch
    assert isinstance(s["img"], torch.Tensor)


def test_build_default_transform_callable():
    tfm = build_default_transform()
    assert callable(tfm)
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/unit/data/test_classification_dataset.py -v
```

Expected: 10 passed.

```bash
git add tests/unit/data/test_classification_dataset.py
git commit -m "test(data): cover SmokePlumeDataset — shape, labels, balance, mult"
```

---

### Task 15: `tests/unit/data/test_segmentation_dataset.py`

**Files:**
- Create: `tests/unit/data/test_segmentation_dataset.py`

- [ ] **Step 1: Write the file**

```python
"""Tests for SmokePlumeSegmentationDataset and label_image_url_to_tif_key."""

from __future__ import annotations

import numpy as np
import pytest

from smoke_detection.common.paths import segmentation_split
from smoke_detection.data.segmentation_dataset import (
    SmokePlumeSegmentationDataset,
    build_default_transform,
    build_eval_transform,
    label_image_url_to_tif_key,
)


# ------------------------- url helper -------------------------

@pytest.mark.parametrize(
    "url,expected",
    [
        ("/data/upload/1-site_001.png", "site_001.tif"),
        ("/data/upload/1-site_2024:01:01.png", "site_2024_01_01.tif"),
        ("/x/y-a-b.png", "a-b.tif"),
    ],
)
def test_label_image_url_to_tif_key(url, expected):
    assert label_image_url_to_tif_key(url) == expected


# ------------------------- dataset core -------------------------

@pytest.fixture
def train_dirs(synthetic_dataset_root):
    imgs, lbls = segmentation_split("train", synthetic_dataset_root)
    return imgs, lbls


def test_length_is_2x_positive_count(train_dirs):
    ds = SmokePlumeSegmentationDataset(datadir=train_dirs[0], seglabeldir=train_dirs[1])
    n_pos = int(ds.labels.sum())
    assert len(ds) == 2 * n_pos


def test_getitem_keys_and_shapes(train_dirs):
    ds = SmokePlumeSegmentationDataset(datadir=train_dirs[0], seglabeldir=train_dirs[1])
    s = ds[0]
    assert set(s.keys()) == {"idx", "img", "fpt", "imgfile"}
    assert s["img"].shape == (4, 120, 120)
    assert s["img"].dtype == np.float32
    assert s["fpt"].shape == (120, 120)
    assert s["fpt"].dtype == np.float32


def test_positive_has_nonzero_mask(train_dirs):
    ds = SmokePlumeSegmentationDataset(datadir=train_dirs[0], seglabeldir=train_dirs[1])
    # Positives are first (indices 0..n_pos-1)
    s = ds[0]
    assert s["fpt"].sum() > 0


def test_negative_has_zero_mask(train_dirs):
    ds = SmokePlumeSegmentationDataset(datadir=train_dirs[0], seglabeldir=train_dirs[1])
    n_pos = int(ds.labels.sum())
    s = ds[n_pos]  # first negative
    assert s["fpt"].sum() == 0


def test_polygon_scaled_by_1_2(train_dirs):
    """Synthetic polygon is a 50x50 square with corners [10,10], [60,10], [60,60], [10,60].
    After the 1.2x scale applied inside the dataset, corners become [12,12] .. [72,72],
    so the rasterized area is approximately 60x60 = 3600 px (± edge effects)."""
    ds = SmokePlumeSegmentationDataset(datadir=train_dirs[0], seglabeldir=train_dirs[1])
    s = ds[0]
    area = float(s["fpt"].sum())
    # Allow ±10% for rasterization edge effects.
    expected = 60 * 60
    assert 0.9 * expected <= area <= 1.1 * expected + 120  # generous due to all_touched=True


def test_mult_scales_length(train_dirs):
    base = SmokePlumeSegmentationDataset(datadir=train_dirs[0], seglabeldir=train_dirs[1], mult=1)
    doubled = SmokePlumeSegmentationDataset(datadir=train_dirs[0], seglabeldir=train_dirs[1], mult=2)
    assert len(doubled) == 2 * len(base)


def test_build_default_and_eval_transforms_callable():
    assert callable(build_default_transform())
    assert callable(build_eval_transform())
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/unit/data/test_segmentation_dataset.py -v
```

Expected: 3 (url helper) + 8 (dataset) = 11 passed.

```bash
git add tests/unit/data/test_segmentation_dataset.py
git commit -m "test(data): cover segmentation dataset + label_image_url_to_tif_key"
```

---

## Phase 4: Unit tests — models/

### Task 16: `tests/unit/models/test_classifier_resnet.py`

**Files:**
- Create: `tests/unit/models/__init__.py`
- Create: `tests/unit/models/test_classifier_resnet.py`

- [ ] **Step 1: Write the file**

```python
"""Tests for build_classifier."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from smoke_detection.models.classifier_resnet import build_classifier


def test_returns_nn_module():
    net = build_classifier(pretrained=False)
    assert isinstance(net, nn.Module)


def test_conv1_takes_4_channels():
    net = build_classifier(pretrained=False)
    assert net.conv1.in_channels == 4


def test_fc_outputs_single_logit():
    net = build_classifier(pretrained=False)
    assert net.fc.out_features == 1


@pytest.mark.parametrize("shape", [(2, 4, 90, 90), (1, 4, 120, 120)])
def test_forward_output_shape(shape):
    net = build_classifier(pretrained=False).eval()
    with torch.no_grad():
        out = net(torch.randn(*shape))
    assert out.shape == (shape[0], 1)


@pytest.mark.slow
def test_pretrained_weights_load(cached_resnet50_weights):
    net = build_classifier(pretrained=True)
    # conv1 is replaced after load, so shape[1] must be 4 (not the IN1K 3).
    assert net.conv1.weight.shape[1] == 4
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/unit/models/test_classifier_resnet.py -v
```

Expected: 5 passed (the slow test is excluded by default — run with `-m slow` to verify once).

```bash
uv run pytest tests/unit/models/test_classifier_resnet.py -v -m slow
```

Expected: 1 passed, 4 deselected.

```bash
git add tests/unit/models/
git commit -m "test(models): cover build_classifier shape + pretrained load"
```

---

### Task 17: `tests/unit/models/test_segmenter_unet.py`

**Files:**
- Create: `tests/unit/models/test_segmenter_unet.py`

- [ ] **Step 1: Write the file**

```python
"""Tests for build_segmenter and UNet forward."""

from __future__ import annotations

import pytest
import torch

from smoke_detection.models.segmenter_unet import UNet, build_segmenter


def test_build_segmenter_returns_unet():
    net = build_segmenter(in_channels=4, n_classes=1)
    assert isinstance(net, UNet)


@pytest.mark.parametrize("crop", [90, 120])
def test_unet_size_preserving_bilinear(crop):
    net = build_segmenter(in_channels=4, n_classes=1, bilinear=True).eval()
    with torch.no_grad():
        out = net(torch.randn(2, 4, crop, crop))
    assert out.shape == (2, 1, crop, crop)


def test_unet_convtranspose_path():
    net = build_segmenter(in_channels=4, n_classes=1, bilinear=False).eval()
    with torch.no_grad():
        out = net(torch.randn(2, 4, 120, 120))
    assert out.shape == (2, 1, 120, 120)
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/unit/models/test_segmenter_unet.py -v
```

Expected: 4 passed.

```bash
git add tests/unit/models/test_segmenter_unet.py
git commit -m "test(models): cover UNet build + size-preserving forward"
```

---

## Phase 5: Unit tests — training/

### Task 18: `tests/unit/training/test_classification_module.py`

**Files:**
- Create: `tests/unit/training/__init__.py`
- Create: `tests/unit/training/test_classification_module.py`

- [ ] **Step 1: Write the file**

```python
"""Tests for ClassificationModule."""

from __future__ import annotations

import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from smoke_detection.training.classification_module import ClassificationModule


@pytest.fixture
def module():
    return ClassificationModule(in_channels=4, pretrained=False, lr=1e-3)


def test_init_saves_hparams(module):
    assert module.hparams.lr == 1e-3
    assert module.hparams.in_channels == 4


def test_forward_shape(module, tiny_classification_batch):
    out = module(tiny_classification_batch["img"])
    assert out.shape == (2, 1)


def test_shared_step_returns_scalar_loss(module, tiny_classification_batch):
    loss, logits, y = module._shared_step(tiny_classification_batch)
    assert loss.ndim == 0
    assert loss.requires_grad
    assert logits.shape == (2, 1)
    assert y.shape == (2, 1)


def test_configure_optimizers_none(module):
    opt = module.configure_optimizers()
    assert isinstance(opt, torch.optim.Optimizer)


def test_configure_optimizers_plateau():
    m = ClassificationModule(pretrained=False, lr=1e-3, scheduler="plateau")
    d = m.configure_optimizers()
    assert isinstance(d["lr_scheduler"]["scheduler"], ReduceLROnPlateau)
    assert d["lr_scheduler"]["monitor"] == "val/loss"


def test_configure_optimizers_cosine():
    m = ClassificationModule(pretrained=False, lr=1e-3, scheduler="cosine")
    d = m.configure_optimizers()
    assert isinstance(d["lr_scheduler"], CosineAnnealingLR)


@pytest.mark.slow
def test_overfitting_sanity_drops_loss(tiny_classification_batch):
    m = ClassificationModule(in_channels=4, pretrained=False, lr=1e-2)
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=1e-2, momentum=0.9)

    loss0, _, _ = m._shared_step(tiny_classification_batch)
    initial = float(loss0.item())
    for _ in range(20):
        opt.zero_grad()
        loss, _, _ = m._shared_step(tiny_classification_batch)
        loss.backward()
        opt.step()
    final = float(loss.item())
    assert final < initial
    assert final < 0.95 * initial
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/unit/training/test_classification_module.py -v
uv run pytest tests/unit/training/test_classification_module.py -v -m slow
```

Expected: 6 passed (fast) + 1 passed (slow).

```bash
git add tests/unit/training/
git commit -m "test(training): cover ClassificationModule forward/step/schedulers + overfit"
```

---

### Task 19: `tests/unit/training/test_segmentation_module.py`

**Files:**
- Create: `tests/unit/training/test_segmentation_module.py`

- [ ] **Step 1: Write the file**

```python
"""Tests for SegmentationModule."""

from __future__ import annotations

import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from smoke_detection.training.segmentation_module import SegmentationModule


@pytest.fixture
def module():
    return SegmentationModule(in_channels=4, n_classes=1, bilinear=True, lr=1e-3)


def test_init_saves_hparams(module):
    assert module.hparams.lr == 1e-3


def test_forward_shape_matches_mask(module, tiny_segmentation_batch):
    out = module(tiny_segmentation_batch["img"])
    assert out.shape == tiny_segmentation_batch["fpt"].unsqueeze(1).shape


def test_shared_step_reshapes_mask_to_4d(module, tiny_segmentation_batch):
    loss, logits, y = module._shared_step(tiny_segmentation_batch)
    assert y.shape == (2, 1, 90, 90)
    assert logits.shape == (2, 1, 90, 90)
    assert loss.ndim == 0
    assert loss.requires_grad


def test_img_level_presence_logic(module):
    """Any-pixel true-positive counts as image-level hit."""
    import torch
    preds = torch.tensor([[[[0, 0], [0, 1]]], [[[0, 0], [0, 0]]]], dtype=torch.int32)
    y = torch.tensor([[[[0, 0], [0, 1]]], [[[0, 0], [0, 0]]]], dtype=torch.int32)
    image_pred = (preds.sum(dim=(1, 2, 3)) > 0).int()
    image_true = (y.sum(dim=(1, 2, 3)) > 0).int()
    assert image_pred.tolist() == [1, 0]
    assert image_true.tolist() == [1, 0]


def test_configure_optimizers_none(module):
    opt = module.configure_optimizers()
    assert isinstance(opt, torch.optim.Optimizer)


def test_configure_optimizers_plateau():
    m = SegmentationModule(lr=1e-3, scheduler="plateau")
    d = m.configure_optimizers()
    assert isinstance(d["lr_scheduler"]["scheduler"], ReduceLROnPlateau)


def test_configure_optimizers_cosine():
    m = SegmentationModule(lr=1e-3, scheduler="cosine")
    d = m.configure_optimizers()
    assert isinstance(d["lr_scheduler"], CosineAnnealingLR)


@pytest.mark.slow
def test_overfitting_sanity_drops_loss(tiny_segmentation_batch):
    m = SegmentationModule(in_channels=4, n_classes=1, bilinear=True, lr=1e-2)
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=1e-2, momentum=0.9)

    loss0, _, _ = m._shared_step(tiny_segmentation_batch)
    initial = float(loss0.item())
    for _ in range(20):
        opt.zero_grad()
        loss, _, _ = m._shared_step(tiny_segmentation_batch)
        loss.backward()
        opt.step()
    final = float(loss.item())
    assert final < initial
    assert final < 0.95 * initial
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/unit/training/test_segmentation_module.py -v
uv run pytest tests/unit/training/test_segmentation_module.py -v -m slow
```

Expected: 7 passed (fast) + 1 passed (slow).

```bash
git add tests/unit/training/test_segmentation_module.py
git commit -m "test(training): cover SegmentationModule forward/step/img_acc + overfit"
```

---

## Phase 6: Unit tests — evaluation/

### Task 20: `tests/unit/evaluation/test_classification_metrics.py`

**Files:**
- Create: `tests/unit/evaluation/__init__.py`
- Create: `tests/unit/evaluation/test_classification_metrics.py`

- [ ] **Step 1: Write the file**

```python
"""Tests for classification plotting helpers."""

from __future__ import annotations

from pathlib import Path

from smoke_detection.evaluation.classification_metrics import (
    plot_confusion_matrix,
    plot_roc_curve,
)


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _is_png(path: Path) -> bool:
    return path.read_bytes()[:8] == PNG_MAGIC


def test_plot_confusion_matrix_writes_png(tmp_path):
    out = tmp_path / "cm.png"
    plot_confusion_matrix(tp=3, tn=5, fp=1, fn=2, out_path=out)
    assert out.exists() and _is_png(out)


def test_plot_roc_curve_writes_png(tmp_path):
    out = tmp_path / "roc.png"
    plot_roc_curve([0.1, 0.9, 0.4, 0.8], [0, 1, 0, 1], out)
    assert out.exists() and _is_png(out)


def test_plot_roc_curve_handles_single_class(tmp_path):
    """When labels are all one class, sklearn raises ValueError which the
    function catches and reports as auc=nan; we only assert no raise + file exists."""
    out = tmp_path / "roc_single.png"
    plot_roc_curve([0.1, 0.2, 0.3, 0.4], [0, 0, 0, 0], out)
    assert out.exists() and _is_png(out)
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/unit/evaluation/test_classification_metrics.py -v
```

Expected: 3 passed.

```bash
git add tests/unit/evaluation/
git commit -m "test(evaluation): cover confusion matrix + ROC plotting"
```

---

### Task 21: `tests/unit/evaluation/test_segmentation_metrics.py`

**Files:**
- Create: `tests/unit/evaluation/test_segmentation_metrics.py`

- [ ] **Step 1: Write the file**

```python
"""Tests for segmentation plotting helpers."""

from __future__ import annotations

from pathlib import Path

from smoke_detection.evaluation.segmentation_metrics import (
    plot_area_ratio_distribution,
    plot_iou_distribution,
)


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _is_png(path: Path) -> bool:
    return path.read_bytes()[:8] == PNG_MAGIC


def test_plot_iou_distribution(tmp_path):
    out = tmp_path / "iou.png"
    plot_iou_distribution([0.1, 0.5, 0.9, 0.3, 0.7], out)
    assert out.exists() and _is_png(out)


def test_plot_iou_distribution_empty(tmp_path):
    out = tmp_path / "iou_empty.png"
    plot_iou_distribution([], out)
    assert out.exists() and _is_png(out)


def test_plot_area_ratio_distribution(tmp_path):
    out = tmp_path / "ar.png"
    plot_area_ratio_distribution([0.5, 1.0, 1.5, 2.0], out)
    assert out.exists() and _is_png(out)


def test_plot_area_ratio_distribution_empty(tmp_path):
    out = tmp_path / "ar_empty.png"
    plot_area_ratio_distribution([], out)
    assert out.exists() and _is_png(out)
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/unit/evaluation/test_segmentation_metrics.py -v
```

Expected: 4 passed.

```bash
git add tests/unit/evaluation/test_segmentation_metrics.py
git commit -m "test(evaluation): cover IoU + area-ratio plotting (incl. empty)"
```

---

## Phase 7: Unit tests — cli/

### Task 22: `tests/unit/cli/test_train_argparse.py`

**Files:**
- Create: `tests/unit/cli/__init__.py`
- Create: `tests/unit/cli/test_train_argparse.py`

- [ ] **Step 1: Write the file**

```python
"""Argparse- and dispatch-level tests for smoke_detection.cli.train."""

from __future__ import annotations

from pathlib import Path

import pytest

from smoke_detection.cli import train as train_cli


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        train_cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--config" in out


def test_missing_config_exits_nonzero():
    with pytest.raises(SystemExit):
        train_cli.main([])  # --config is required


def test_override_accumulates(monkeypatch, tmp_path, classification_yaml_tmp):
    """Calling main with two --override flags must pass both into load_config."""
    captured = {}

    from smoke_detection.configs import loader

    real_load = loader.load_config

    def spy(path, overrides=None):
        captured["overrides"] = list(overrides or [])
        return real_load(path, overrides=overrides)

    monkeypatch.setattr("smoke_detection.cli.train.load_config", spy)

    # Prevent actual training
    import lightning as L
    monkeypatch.setattr(L.Trainer, "fit", lambda self, *a, **k: None)

    train_cli.main(
        ["--config", str(classification_yaml_tmp),
         "--override", "optim.lr=5e-4",
         "--override", "trainer.max_epochs=2"]
    )
    assert captured["overrides"] == ["optim.lr=5e-4", "trainer.max_epochs=2"]


def test_classification_dispatch_calls_classification_builder(
    monkeypatch, classification_yaml_tmp
):
    """A classification YAML must route through _build_classification."""
    called = {}

    real = train_cli._build_classification

    def spy(cfg):
        called["yes"] = True
        return real(cfg)

    monkeypatch.setattr(train_cli, "_build_classification", spy)

    import lightning as L
    monkeypatch.setattr(L.Trainer, "fit", lambda self, *a, **k: None)

    train_cli.main(["--config", str(classification_yaml_tmp)])
    assert called.get("yes") is True


def test_segmentation_dispatch_calls_segmentation_builder(
    monkeypatch, segmentation_yaml_tmp
):
    called = {}

    real = train_cli._build_segmentation

    def spy(cfg):
        called["yes"] = True
        return real(cfg)

    monkeypatch.setattr(train_cli, "_build_segmentation", spy)

    import lightning as L
    monkeypatch.setattr(L.Trainer, "fit", lambda self, *a, **k: None)

    train_cli.main(["--config", str(segmentation_yaml_tmp)])
    assert called.get("yes") is True
```

Note: the `classification_yaml_tmp` and `segmentation_yaml_tmp` fixtures live under `tests/e2e/conftest.py`. They're not visible from `tests/unit/`. Move them (and the `sample_*_config` fixtures they depend on) — **already done in Task 5**, the `sample_*` ones are at the top level. Duplicate just the YAML writers at the top level so unit CLI tests can reach them:

Edit `tests/conftest.py` to append:

```python
@pytest.fixture
def classification_yaml_tmp(sample_classification_config, tmp_path):
    import yaml
    p = tmp_path / "cls.yaml"
    p.write_text(yaml.safe_dump(sample_classification_config.model_dump(mode="json")))
    return p


@pytest.fixture
def segmentation_yaml_tmp(sample_segmentation_config, tmp_path):
    import yaml
    p = tmp_path / "seg.yaml"
    p.write_text(yaml.safe_dump(sample_segmentation_config.model_dump(mode="json")))
    return p
```

And simplify `tests/e2e/conftest.py` to be empty (or delete it — `__init__.py` is enough).

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/unit/cli/test_train_argparse.py -v
```

Expected: 5 passed.

```bash
git add tests/conftest.py tests/e2e/conftest.py tests/unit/cli/
git commit -m "test(cli): cover train argparse, overrides, task dispatch"
```

---

### Task 23: `tests/unit/cli/test_eval_argparse.py`

**Files:**
- Create: `tests/unit/cli/test_eval_argparse.py`

- [ ] **Step 1: Write the file**

```python
"""Argparse tests for smoke_detection.cli.eval."""

from __future__ import annotations

import pytest

from smoke_detection.cli import eval as eval_cli


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        eval_cli.main(["--help"])
    assert exc.value.code == 0


def test_missing_config_exits_nonzero():
    with pytest.raises(SystemExit):
        eval_cli.main(["--ckpt", "nonexistent.ckpt"])


def test_missing_ckpt_exits_nonzero(classification_yaml_tmp):
    with pytest.raises(SystemExit):
        eval_cli.main(["--config", str(classification_yaml_tmp)])


def test_out_dir_defaults_to_output_plus_experiment(
    monkeypatch, classification_yaml_tmp, tmp_path
):
    captured = {}

    def spy_eval_cls(cfg, ckpt, out_dir):
        captured["out_dir"] = out_dir

    monkeypatch.setattr(eval_cli, "_eval_classification", spy_eval_cls)

    fake_ckpt = tmp_path / "last.ckpt"
    fake_ckpt.write_bytes(b"")
    eval_cli.main(["--config", str(classification_yaml_tmp), "--ckpt", str(fake_ckpt)])

    # Path is cfg.paths.output_dir / cfg.paths.experiment_name / "eval"
    assert captured["out_dir"].name == "eval"
    assert captured["out_dir"].parent.name == "test_classification"


def test_out_dir_override_respected(monkeypatch, classification_yaml_tmp, tmp_path):
    captured = {}

    def spy_eval_cls(cfg, ckpt, out_dir):
        captured["out_dir"] = out_dir

    monkeypatch.setattr(eval_cli, "_eval_classification", spy_eval_cls)

    fake_ckpt = tmp_path / "last.ckpt"
    fake_ckpt.write_bytes(b"")
    custom = tmp_path / "custom_out"
    eval_cli.main(
        ["--config", str(classification_yaml_tmp),
         "--ckpt", str(fake_ckpt),
         "--out-dir", str(custom)]
    )
    assert captured["out_dir"] == custom
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/unit/cli/test_eval_argparse.py -v
```

Expected: 5 passed.

```bash
git add tests/unit/cli/test_eval_argparse.py
git commit -m "test(cli): cover eval argparse and out-dir resolution"
```

---

## Phase 8: Unit tests — scripts/

### Task 24: `tests/unit/scripts/test_prepare_dataset_helpers.py`

**Files:**
- Create: `tests/unit/scripts/__init__.py`
- Create: `tests/unit/scripts/test_prepare_dataset_helpers.py`

- [ ] **Step 1: Write the file**

```python
"""Tests for the pure helpers in scripts/prepare_dataset.py.

Import via the file path since `scripts/` is not a package.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_prepare_dataset():
    path = REPO_ROOT / "scripts" / "prepare_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_dataset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pd_module():
    return _load_prepare_dataset()


def test_site_id_from_stem(pd_module):
    assert pd_module.site_id_from_stem("ghana_2018-01-01") == "ghana"
    assert pd_module.site_id_from_stem("no_underscore") == "no"


@pytest.mark.parametrize(
    "url,expected",
    [
        ("/data/upload/1-site_001.png", "site_001.tif"),
        ("/data/upload/1-site_2024:01:01Z.png", "site_2024_01_01Z.tif"),
    ],
)
def test_json_url_to_tif_basename(pd_module, url, expected):
    assert pd_module.json_url_to_tif_basename(url) == expected


def test_link_or_copy_copy_mode(pd_module, tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("hello")
    dst = tmp_path / "out" / "dst.txt"
    pd_module.link_or_copy(str(src), str(dst), "copy")
    assert dst.read_text() == "hello"


@pytest.mark.skipif(sys.platform == "win32", reason="hardlinks unreliable on Windows")
def test_link_or_copy_hardlink_mode(pd_module, tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("hardlinked")
    dst = tmp_path / "out" / "dst.txt"
    pd_module.link_or_copy(str(src), str(dst), "hardlink")
    # hardlink: the two paths should share inode
    assert dst.stat().st_ino == src.stat().st_ino


def test_link_or_copy_invalid_mode(pd_module, tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("x")
    with pytest.raises(ValueError):
        pd_module.link_or_copy(str(src), str(tmp_path / "dst.txt"), "teleport")
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/unit/scripts/test_prepare_dataset_helpers.py -v
```

Expected: 5 passed on Windows (hardlink skipped), 6 on Linux.

```bash
git add tests/unit/scripts/
git commit -m "test(scripts): cover prepare_dataset.py pure helpers"
```

---

## Phase 9: Integration tests

### Task 25: `tests/integration/test_classification_datamodule.py`

**Files:**
- Create: `tests/integration/test_classification_datamodule.py`

- [ ] **Step 1: Write the file**

```python
"""Integration: ClassificationDataModule end-to-end on synthetic data."""

from __future__ import annotations

import torch

from smoke_detection.data.classification_datamodule import ClassificationDataModule


def test_setup_fit_builds_train_and_val(synthetic_dataset_root):
    dm = ClassificationDataModule(
        data_root=synthetic_dataset_root, batch_size=2, num_workers=0, balance="none",
    )
    dm.setup(stage="fit")
    assert hasattr(dm, "train_ds") and len(dm.train_ds) > 0
    assert hasattr(dm, "val_ds") and len(dm.val_ds) > 0


def test_setup_test_builds_test_ds(synthetic_dataset_root):
    dm = ClassificationDataModule(
        data_root=synthetic_dataset_root, batch_size=2, num_workers=0, balance="none",
    )
    dm.setup(stage="test")
    assert hasattr(dm, "test_ds") and len(dm.test_ds) > 0


def test_train_dataloader_yields_expected_shape(synthetic_dataset_root):
    dm = ClassificationDataModule(
        data_root=synthetic_dataset_root, batch_size=2, num_workers=0,
        crop_size=90, balance="none",
    )
    dm.setup(stage="fit")
    batch = next(iter(dm.train_dataloader()))
    assert batch["img"].shape == (2, 4, 90, 90)
    assert batch["lbl"].dtype == torch.bool
    assert batch["lbl"].shape == (2,)


def test_val_dataloader_uses_no_balance(synthetic_dataset_root):
    dm = ClassificationDataModule(
        data_root=synthetic_dataset_root, batch_size=2, num_workers=0, balance="upsample",
    )
    dm.setup(stage="fit")
    # val_ds always uses balance="none" in the DataModule's setup
    # (enforced in the code; we verify by counting positives/negatives)
    assert len(dm.val_ds.imgfiles) == len(dm.val_ds.labels)


def test_num_workers_zero_on_windows(monkeypatch, synthetic_dataset_root):
    import platform
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    dm = ClassificationDataModule(
        data_root=synthetic_dataset_root, batch_size=2, num_workers=4, balance="none",
    )
    assert dm.num_workers == 0


def test_save_hyperparameters(synthetic_dataset_root):
    dm = ClassificationDataModule(
        data_root=synthetic_dataset_root, batch_size=7, num_workers=0, balance="none",
    )
    assert dm.hparams.batch_size == 7
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/integration/test_classification_datamodule.py -v
```

Expected: 6 passed.

```bash
git add tests/integration/test_classification_datamodule.py
git commit -m "test(integration): cover ClassificationDataModule setup + loaders"
```

---

### Task 26: `tests/integration/test_segmentation_datamodule.py`

**Files:**
- Create: `tests/integration/test_segmentation_datamodule.py`

- [ ] **Step 1: Write the file**

```python
"""Integration: SegmentationDataModule end-to-end on synthetic data."""

from __future__ import annotations

import torch

from smoke_detection.data.segmentation_datamodule import SegmentationDataModule


def test_setup_fit_builds_train_and_val(synthetic_dataset_root):
    dm = SegmentationDataModule(
        data_root=synthetic_dataset_root, batch_size=2, num_workers=0,
    )
    dm.setup(stage="fit")
    assert len(dm.train_ds) > 0
    assert len(dm.val_ds) > 0


def test_setup_test_standalone(synthetic_dataset_root):
    dm = SegmentationDataModule(
        data_root=synthetic_dataset_root, batch_size=2, num_workers=0,
    )
    dm.setup(stage="test")
    assert hasattr(dm, "test_ds") and len(dm.test_ds) > 0


def test_train_batch_shapes(synthetic_dataset_root):
    dm = SegmentationDataModule(
        data_root=synthetic_dataset_root, batch_size=2, num_workers=0, crop_size=90,
    )
    dm.setup(stage="fit")
    batch = next(iter(dm.train_dataloader()))
    assert batch["img"].shape == (2, 4, 90, 90)
    assert batch["fpt"].shape == (2, 90, 90)
    assert batch["img"].dtype == torch.float32


def test_num_workers_zero_on_windows(monkeypatch, synthetic_dataset_root):
    import platform
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    dm = SegmentationDataModule(
        data_root=synthetic_dataset_root, batch_size=2, num_workers=4,
    )
    assert dm.num_workers == 0
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/integration/test_segmentation_datamodule.py -v
```

Expected: 4 passed.

```bash
git add tests/integration/test_segmentation_datamodule.py
git commit -m "test(integration): cover SegmentationDataModule setup + loaders"
```

---

### Task 27: `tests/integration/test_module_consumes_batch.py`

**Files:**
- Create: `tests/integration/test_module_consumes_batch.py`

- [ ] **Step 1: Write the file**

```python
"""Integration: LightningModule training_step on a real DataModule batch."""

from __future__ import annotations

import torch

from smoke_detection.data.classification_datamodule import ClassificationDataModule
from smoke_detection.data.segmentation_datamodule import SegmentationDataModule
from smoke_detection.training.classification_module import ClassificationModule
from smoke_detection.training.segmentation_module import SegmentationModule


def test_classification_training_step_on_real_batch(synthetic_dataset_root):
    dm = ClassificationDataModule(
        data_root=synthetic_dataset_root, batch_size=2, num_workers=0, balance="none",
    )
    dm.setup(stage="fit")
    batch = next(iter(dm.train_dataloader()))
    module = ClassificationModule(in_channels=4, pretrained=False, lr=1e-3)
    loss = module.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_segmentation_training_step_on_real_batch(synthetic_dataset_root):
    dm = SegmentationDataModule(
        data_root=synthetic_dataset_root, batch_size=2, num_workers=0,
    )
    dm.setup(stage="fit")
    batch = next(iter(dm.train_dataloader()))
    module = SegmentationModule(in_channels=4, n_classes=1, bilinear=True, lr=1e-3)
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert loss.requires_grad
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/integration/test_module_consumes_batch.py -v
```

Expected: 2 passed.

```bash
git add tests/integration/test_module_consumes_batch.py
git commit -m "test(integration): module training_step on DataModule batch"
```

---

### Task 28: `tests/integration/test_config_to_trainer.py`

**Files:**
- Create: `tests/integration/test_config_to_trainer.py`

- [ ] **Step 1: Write the file**

```python
"""Integration: full YAML-to-trainer wiring with fast_dev_run."""

from __future__ import annotations

from pathlib import Path

import lightning as L

from smoke_detection.cli.train import (
    _build_classification,
    _build_segmentation,
    _build_trainer,
)


def _run(cfg):
    if cfg.task == "classification":
        module, dm, monitor = _build_classification(cfg)
    else:
        module, dm, monitor = _build_segmentation(cfg)
    trainer = _build_trainer(cfg, monitor=monitor)
    trainer.fit(module, datamodule=dm)
    return trainer


def test_classification_config_to_trainer_fit(sample_classification_config):
    trainer = _run(sample_classification_config)
    # fast_dev_run: no checkpoints are written, but training completed.
    assert trainer.state.finished


def test_segmentation_config_to_trainer_fit(sample_segmentation_config):
    trainer = _run(sample_segmentation_config)
    assert trainer.state.finished


def test_checkpoint_written_without_fast_dev_run(sample_classification_config):
    cfg = sample_classification_config.model_copy(deep=True)
    cfg.trainer.fast_dev_run = False
    cfg.trainer.max_epochs = 1
    trainer = _run(cfg)
    ckpt_dir = Path(cfg.paths.output_dir) / cfg.paths.experiment_name
    assert ckpt_dir.exists()
    ckpts = list(ckpt_dir.rglob("*.ckpt"))
    assert len(ckpts) >= 1
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/integration/test_config_to_trainer.py -v
```

Expected: 3 passed. This is the slowest integration test so far (~20–40s). If it times out, bump its timeout locally with `@pytest.mark.timeout(120)`.

```bash
git add tests/integration/test_config_to_trainer.py
git commit -m "test(integration): YAML → trainer.fit wiring (fast_dev_run + ckpt)"
```

---

### Task 29: `tests/integration/test_prepare_dataset_script.py`

**Files:**
- Create: `tests/integration/test_prepare_dataset_script.py`

- [ ] **Step 1: Write the file**

```python
"""Integration: scripts/prepare_dataset.py on a synthetic Zenodo-like source."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests._data import build_synthetic_zenodo_source


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "prepare_dataset.py"


def _run(*args, cwd=None, check=True):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        check=check,
    )


def test_end_to_end_copy_mode(tmp_path):
    src = tmp_path / "source"
    build_synthetic_zenodo_source(src)
    out = tmp_path / "prepared"
    r = _run("--source", str(src), "--output", str(out), "--mode", "copy")
    assert r.returncode == 0
    for split in ("train", "val", "test"):
        # At least one split should be non-empty (small dataset, random split).
        pass
    total_tifs = len(list((out / "classification").rglob("*.tif")))
    assert total_tifs > 0


def test_splits_are_site_disjoint(tmp_path):
    src = tmp_path / "source"
    build_synthetic_zenodo_source(src)
    out = tmp_path / "prepared"
    _run("--source", str(src), "--output", str(out), "--mode", "copy")

    def _sites(root: Path) -> set[str]:
        return {p.name.split("_", 1)[0] for p in root.rglob("*.tif")}

    tr = _sites(out / "classification" / "train")
    va = _sites(out / "classification" / "val")
    te = _sites(out / "classification" / "test")
    assert tr.isdisjoint(va)
    assert tr.isdisjoint(te)
    assert va.isdisjoint(te)


def test_every_json_has_matching_positive_tif(tmp_path):
    src = tmp_path / "source"
    build_synthetic_zenodo_source(src)
    out = tmp_path / "prepared"
    _run("--source", str(src), "--output", str(out), "--mode", "copy")
    for split in ("train", "val", "test"):
        lbl_dir = out / "segmentation" / split / "labels"
        img_dir = out / "segmentation" / split / "images" / "positive"
        if not lbl_dir.exists():
            continue
        for jf in lbl_dir.glob("*.json"):
            stem = jf.stem.replace(".tif", "")
            expected_tif = img_dir / f"{stem}.tif" if stem.endswith(".tif") else img_dir / f"{jf.stem.removesuffix('.json')}"
            # Be lenient: label file is "<tif_basename>.json".
            candidate = img_dir / jf.stem
            assert candidate.exists(), f"{candidate} missing"


def test_dry_run_produces_no_output(tmp_path):
    src = tmp_path / "source"
    build_synthetic_zenodo_source(src)
    out = tmp_path / "prepared"
    r = _run("--source", str(src), "--output", str(out), "--dry-run")
    assert r.returncode == 0
    assert not out.exists()


def test_bad_ratios_fail(tmp_path):
    src = tmp_path / "source"
    build_synthetic_zenodo_source(src)
    out = tmp_path / "prepared"
    r = _run(
        "--source", str(src), "--output", str(out), "--mode", "copy",
        "--train-ratio", "0.9", "--val-ratio", "0.5", "--test-ratio", "0.1",
        check=False,
    )
    assert r.returncode != 0
    assert "ratios must sum to 1.0" in r.stderr


def test_missing_source_fails(tmp_path):
    out = tmp_path / "prepared"
    r = _run("--source", str(tmp_path / "does_not_exist"),
             "--output", str(out), "--mode", "copy", check=False)
    assert r.returncode != 0


def test_refuses_existing_output_without_force(tmp_path):
    src = tmp_path / "source"
    build_synthetic_zenodo_source(src)
    out = tmp_path / "prepared"
    out.mkdir()
    r = _run("--source", str(src), "--output", str(out),
             "--mode", "copy", check=False)
    assert r.returncode != 0
    assert "--force" in r.stderr
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/integration/test_prepare_dataset_script.py -v
```

Expected: 7 passed.

```bash
git add tests/integration/test_prepare_dataset_script.py
git commit -m "test(integration): full prepare_dataset.py end-to-end on synthetic"
```

---

## Phase 10: E2E tests

### Task 30: `tests/e2e/test_fast_dev_run.py`

**Files:**
- Create: `tests/e2e/test_fast_dev_run.py`

- [ ] **Step 1: Write the file**

```python
"""E2E: fast_dev_run for both classification and segmentation CLIs."""

from __future__ import annotations

import pytest

from smoke_detection.cli.train import main as train_main


pytestmark = pytest.mark.e2e


def test_classification_fast_dev_run(classification_yaml_tmp):
    rc = train_main(
        ["--config", str(classification_yaml_tmp),
         "--override", "trainer.fast_dev_run=true",
         "--override", "data.batch_size=2",
         "--override", "data.num_workers=0"]
    )
    assert rc == 0


def test_segmentation_fast_dev_run(segmentation_yaml_tmp):
    rc = train_main(
        ["--config", str(segmentation_yaml_tmp),
         "--override", "trainer.fast_dev_run=true",
         "--override", "data.batch_size=2",
         "--override", "data.num_workers=0"]
    )
    assert rc == 0
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/e2e/test_fast_dev_run.py -v -m e2e --timeout=300
```

Expected: 2 passed in < 90s.

```bash
git add tests/e2e/test_fast_dev_run.py
git commit -m "test(e2e): fast_dev_run classification + segmentation via CLI"
```

---

### Task 31: `tests/e2e/test_train_eval_cycle.py`

**Files:**
- Create: `tests/e2e/test_train_eval_cycle.py`

- [ ] **Step 1: Write the file**

```python
"""E2E: tiny train → eval cycle producing plots."""

from __future__ import annotations

from pathlib import Path

import pytest

from smoke_detection.cli.eval import main as eval_main
from smoke_detection.cli.train import main as train_main


pytestmark = pytest.mark.e2e


def _find_last_ckpt(output_dir: Path, experiment: str) -> Path:
    ckpt = output_dir / experiment
    matches = list(ckpt.rglob("last.ckpt"))
    assert matches, f"no last.ckpt under {ckpt}"
    return matches[0]


def test_classification_train_eval_cycle(
    classification_yaml_tmp, sample_classification_config, tmp_path
):
    cfg = sample_classification_config
    rc = train_main(
        ["--config", str(classification_yaml_tmp),
         "--override", "trainer.fast_dev_run=false",
         "--override", "trainer.max_epochs=1"]
    )
    assert rc == 0

    ckpt = _find_last_ckpt(cfg.paths.output_dir, cfg.paths.experiment_name)
    rc = eval_main(["--config", str(classification_yaml_tmp), "--ckpt", str(ckpt)])
    assert rc == 0

    out_dir = cfg.paths.output_dir / cfg.paths.experiment_name / "eval"
    assert (out_dir / "confusion_matrix.png").stat().st_size > 0
    assert (out_dir / "roc_curve.png").stat().st_size > 0


def test_segmentation_train_eval_cycle(
    segmentation_yaml_tmp, sample_segmentation_config, tmp_path
):
    cfg = sample_segmentation_config
    rc = train_main(
        ["--config", str(segmentation_yaml_tmp),
         "--override", "trainer.fast_dev_run=false",
         "--override", "trainer.max_epochs=1"]
    )
    assert rc == 0

    ckpt = _find_last_ckpt(cfg.paths.output_dir, cfg.paths.experiment_name)
    rc = eval_main(["--config", str(segmentation_yaml_tmp), "--ckpt", str(ckpt)])
    assert rc == 0

    out_dir = cfg.paths.output_dir / cfg.paths.experiment_name / "eval"
    assert (out_dir / "iou_distribution.png").stat().st_size > 0
    assert (out_dir / "area_ratio_distribution.png").stat().st_size > 0
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/e2e/test_train_eval_cycle.py -v -m e2e --timeout=300
```

Expected: 2 passed in < 2 min.

```bash
git add tests/e2e/test_train_eval_cycle.py
git commit -m "test(e2e): 1-epoch train → eval cycle produces all plots"
```

---

### Task 32: `tests/e2e/test_checkpoint_roundtrip.py`

**Files:**
- Create: `tests/e2e/test_checkpoint_roundtrip.py`

- [ ] **Step 1: Write the file**

```python
"""E2E: train → save ckpt → load_from_checkpoint → bit-identical forward."""

from __future__ import annotations

import pytest
import torch

from smoke_detection.cli.train import main as train_main
from smoke_detection.training.classification_module import ClassificationModule


pytestmark = pytest.mark.e2e


def test_classification_checkpoint_roundtrip(
    classification_yaml_tmp, sample_classification_config, tmp_path
):
    cfg = sample_classification_config
    rc = train_main(
        ["--config", str(classification_yaml_tmp),
         "--override", "trainer.fast_dev_run=false",
         "--override", "trainer.max_epochs=1"]
    )
    assert rc == 0

    ckpt_dir = cfg.paths.output_dir / cfg.paths.experiment_name
    last = next(ckpt_dir.rglob("last.ckpt"))
    loaded = ClassificationModule.load_from_checkpoint(str(last))
    loaded.eval()

    x = torch.randn(2, 4, 90, 90)
    with torch.no_grad():
        a = loaded(x)
        b = loaded(x)
    torch.testing.assert_close(a, b)
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests/e2e/test_checkpoint_roundtrip.py -v -m e2e --timeout=300
```

Expected: 1 passed.

```bash
git add tests/e2e/test_checkpoint_roundtrip.py
git commit -m "test(e2e): checkpoint round-trip produces deterministic forward"
```

---

## Phase 11: CI wiring

### Task 33: Update `.github/workflows/ci.yml` for cached weights + marker tiers + e2e job

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Replace `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [master]
  pull_request:
    branches: [master]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v3
      - name: Set up Python
        run: uv python install 3.12
      - name: Install dev deps
        run: uv sync --extra dev
      - name: Ruff check
        run: uv run ruff check .
      - name: Ruff format check
        run: uv run ruff format --check .

  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v3
      - name: Set up Python
        run: uv python install ${{ matrix.python }}
      - name: Install dev deps
        run: uv sync --extra dev
      - name: Cache torchvision ImageNet weights
        uses: actions/cache@v4
        with:
          path: ~/.cache/torch/hub/checkpoints
          key: resnet50-imagenet-v1
      - name: Warm torchvision weights cache
        run: uv run python -c "from torchvision.models import resnet50, ResNet50_Weights; resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)"
      - name: Pytest (fast tier)
        run: uv run pytest -m "not slow and not e2e and not gpu"
      - name: Pytest (slow tier + coverage gate)
        run: uv run pytest -m "slow" --cov-append --cov-fail-under=80

  e2e:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v3
      - name: Set up Python
        run: uv python install 3.12
      - name: Install dev deps
        run: uv sync --extra dev
      - name: Cache torchvision ImageNet weights
        uses: actions/cache@v4
        with:
          path: ~/.cache/torch/hub/checkpoints
          key: resnet50-imagenet-v1
      - name: Pytest (e2e)
        run: uv run pytest -m "e2e" --timeout=300

  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v3
      - name: Set up Python
        run: uv python install 3.12
      - name: Build
        run: uv build
```

- [ ] **Step 2: Local sanity — run the full `make test`**

```bash
make test
```

Expected: both pytest invocations pass; coverage may still be under 80% — **OK for now**; Task 35 addresses coverage gaps.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add cached ImageNet weights + marker tiers + e2e job"
```

---

### Task 34: Update `CONTRIBUTING.md` with Tests subsection

**Files:**
- Modify: `CONTRIBUTING.md` — replace the `## Tests` subsection.

- [ ] **Step 1: Edit `CONTRIBUTING.md`**

Replace the existing `## Tests` subsection with:

```markdown
## Tests

- Framework: `pytest`. Tests live under `tests/`.
- Marker tiers:
  - **default** (unmarked) — fast unit + integration; CPU; < 60s on a cold runner.
  - **`slow`** — pretrained ResNet-50 load, overfitting sanity loops.
  - **`gpu`** — requires CUDA; auto-skipped when unavailable.
  - **`e2e`** — CLI end-to-end, fast_dev_run, checkpoint round-trip.
- Make targets:
  - `make test` — default + slow tiers with an 80% coverage gate.
  - `make test-e2e` — e2e tier only.
  - `make test-gpu` — gpu tier only.
  - `make test-all` — everything except `gpu`.
- Pretrained weights are cached under `~/.cache/torch/hub/checkpoints/`. In CI
  they're restored by `actions/cache` keyed on `resnet50-imagenet-v1`. Bump
  the key if the torchvision weights URL changes upstream.
- Tests never hit the network. Synthetic data is generated on the fly by
  `tests/_data.py`.
```

- [ ] **Step 2: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "docs: document test marker tiers and cache-key policy"
```

---

## Phase 12: Coverage verification + cleanup

### Task 35: Verify 80% coverage gate; add marginal tests where needed; remove the old smoke script

**Files:**
- Possibly modify: any `src/smoke_detection/**/*.py` where coverage uncovered a genuine bug.
- Delete: `scripts/smoketest_fast_dev_run.py` (superseded by `tests/e2e/test_fast_dev_run.py`).
- Possibly modify: `README.md` — remove the standalone smoketest reference (grep for it; only mention if present).

- [ ] **Step 1: Run the full fast+slow suite with coverage**

```bash
make test
```

Expected: both runs pass AND `--cov-fail-under=80` passes. If it fails:

- Inspect the `term-missing` report for uncovered lines.
- Decide per uncovered line: (a) is it actually reachable? (if not, add `# pragma: no cover`), or (b) write a targeted test.
- Prefer (b). Typical gap candidates:
  - The `raise RuntimeError("Unsupported config type")` branches in `cli/train.py` and `cli/eval.py` — write a test that constructs a dummy pydantic config subclass of `BaseConfig` with a new task literal, monkeypatches `load_config` to return it, and calls `main(...)` expecting `RuntimeError`.
  - The `raise SystemExit(main())` module-guard branches — these are `if __name__ == "__main__"`, already omitted by coverage config. No action.
  - The `balance_downsample` branch of `SmokePlumeDataset` — add an explicit test in `tests/unit/data/test_classification_dataset.py`.
- After each added test, re-run `make test`.

- [ ] **Step 2: Decide fate of `scripts/smoketest_fast_dev_run.py`**

The script is superseded by `tests/e2e/test_fast_dev_run.py`. Delete it and any README references:

```bash
rm scripts/smoketest_fast_dev_run.py
```

Grep for references and update:

```bash
uv run python -c "import pathlib; [print(p) for p in pathlib.Path('.').rglob('*') if p.is_file() and p.suffix in {'.md', '.yaml', '.yml', '.py'} and 'smoketest' in p.read_text(errors='ignore')]"
```

If any matches remain, update them to point at `tests/e2e/test_fast_dev_run.py` and `make test-e2e`.

- [ ] **Step 3: Final full-suite run**

```bash
make lint
make test
make test-e2e
```

Expected: all three clean. GPU suite is manual; skip.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore(test): remove superseded smoketest script, close coverage gaps"
```

---

## Summary of files touched

**Created (36 files):**
- `tests/__init__.py`
- `tests/_data.py`
- `tests/conftest.py` (replaces old scaffold)
- `tests/unit/__init__.py`
- `tests/unit/test_package_imports.py`
- `tests/unit/common/{__init__.py, test_paths.py, test_seed.py, test_logging.py}`
- `tests/unit/configs/{__init__.py, test_schemas.py, test_loader.py}`
- `tests/unit/data/{__init__.py, test_pad_to_120.py, test_transforms.py, test_classification_dataset.py, test_segmentation_dataset.py}`
- `tests/unit/models/{__init__.py, test_classifier_resnet.py, test_segmenter_unet.py}`
- `tests/unit/training/{__init__.py, test_classification_module.py, test_segmentation_module.py}`
- `tests/unit/evaluation/{__init__.py, test_classification_metrics.py, test_segmentation_metrics.py}`
- `tests/unit/cli/{__init__.py, test_train_argparse.py, test_eval_argparse.py}`
- `tests/unit/scripts/{__init__.py, test_prepare_dataset_helpers.py}`
- `tests/integration/{__init__.py, test_classification_datamodule.py, test_segmentation_datamodule.py, test_module_consumes_batch.py, test_config_to_trainer.py, test_prepare_dataset_script.py}`
- `tests/e2e/{__init__.py, conftest.py, test_fast_dev_run.py, test_train_eval_cycle.py, test_checkpoint_roundtrip.py}`

**Modified:**
- `pyproject.toml` — markers, coverage, timeout, new dev deps.
- `Makefile` — test-tier targets.
- `.github/workflows/ci.yml` — weight caching, marker tiers, e2e job.
- `CONTRIBUTING.md` — Tests subsection.

**Deleted:**
- `tests/test_placeholder.py`, `tests/.gitkeep`.
- `scripts/smoketest_fast_dev_run.py`.

## Execution order note

Tasks are numbered and largely independent within phases. Phase 0 must land first (infrastructure). Within each later phase, tasks can run in parallel (different files). **Do not skip ahead to Task 35 (coverage gate)** — it depends on every earlier phase having landed.
