# Repo Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the Industrial Smoke Plume Detection repo into a standard Python + ML project layout, port training/eval to PyTorch Lightning, and introduce pydantic + YAML experiment configs.

**Architecture:** Three layers of change: (1) repository hygiene (CI, linting, pre-commit, Makefile, deps consolidation); (2) package reorganization into `common/ data/ models/ training/ evaluation/ configs/ cli/` with pure `nn.Module`s separated from `LightningModule`s and `LightningDataModule`s; (3) config-driven CLI (`--config path/to.yaml`) replacing the previous argparse-per-script scheme.

**Tech Stack:** Python 3.11+/3.12 (dev), PyTorch 2.2+, Lightning 2.2+, torchmetrics, pydantic v2 + pydantic-settings, YAML, `uv` for installs, `ruff` (lint+format), `pytest`, `pre-commit`, GitHub Actions.

**Reference spec:** `docs/superpowers/specs/2026-04-17-repo-cleanup-design.md`

**Working directory:** All paths in this plan are relative to the repo root `C:/Users/kampw/PycharmProjects/IndustrialSmokePlumeDetection/`.

---

## Phase 1 — Scaffolding & Hygiene

### Task 1: Fix `pyproject.toml` and consolidate deps

**Files:**
- Modify: `pyproject.toml`
- Delete: `requirements.txt`

- [ ] **Step 1: Rewrite `pyproject.toml`**

Replace the entire file contents with:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "industrial-smoke-plume-detection"
version = "0.2.0"
description = "Industrial smoke plume detection and segmentation from Sentinel-2 imagery"
readme = "README.md"
license = { file = "LICENSE" }
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.24",
    "matplotlib>=3.7",
    "rasterio>=1.3",
    "scikit-learn>=1.3",
    "shapely>=2.0",
    "tqdm>=4.65",
    "tensorboard>=2.14",
    "torch>=2.2",
    "torchvision>=0.17",
    "lightning>=2.2",
    "torchmetrics>=1.3",
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.4",
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "pre-commit>=3.6",
]
notebooks = [
    "jupyter>=1.0",
    "ipykernel>=6.29",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "SIM"]
ignore = ["E501"]  # line-length handled by formatter

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
```

- [ ] **Step 2: Delete `requirements.txt`**

```bash
rm requirements.txt
```

- [ ] **Step 3: Verify `uv sync` succeeds**

```bash
uv sync --extra dev
```

Expected: resolves all dependencies without error. `lightning`, `torchmetrics`, `pydantic-settings`, `ruff`, `pytest` appear in the environment.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml requirements.txt uv.lock
git commit -m "chore: fix build-backend, consolidate deps, add Lightning/pydantic/dev tooling"
```

---

### Task 2: Pin Python version and expand `.gitignore`

**Files:**
- Create: `.python-version`
- Modify: `.gitignore`

- [ ] **Step 1: Write `.python-version`**

File contents:

```
3.12
```

- [ ] **Step 2: Expand `.gitignore`**

Append the following block to the end of the existing `.gitignore`:

```gitignore

# --- ML artefacts ---
lightning_logs/
mlruns/
wandb/
*.ckpt

# --- Project data conventions ---
data/*
!data/.gitkeep

# --- Notebook outputs ---
notebooks/*
!notebooks/.gitkeep
```

(Leave existing rules untouched; the `*.model`, `runs/`, `outputs/`, and `4250706/` rules stay.)

- [ ] **Step 3: Commit**

```bash
git add .python-version .gitignore
git commit -m "chore: pin Python 3.12 for dev, extend .gitignore for Lightning/data/notebooks"
```

---

### Task 3: Add `pre-commit` config

**Files:**
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Write `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ["--maxkb=1024"]

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.8
    hooks:
      - id: ruff
        args: ["--fix"]
      - id: ruff-format
```

- [ ] **Step 2: Install the hooks locally (optional smoke-check)**

```bash
uv run pre-commit install
uv run pre-commit run --all-files || true
```

Expected: may report formatting issues on existing files (that's fine — they'll be reformatted in Task 4 or later cleanup).

- [ ] **Step 3: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "chore: add pre-commit config (ruff, hygiene hooks)"
```

---

### Task 4: Add `Makefile` shortcuts

**Files:**
- Create: `Makefile`

- [ ] **Step 1: Write `Makefile`**

```makefile
.PHONY: install lint format test clean train-cls train-seg eval-cls eval-seg

install:
	uv sync --extra dev

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest

CONFIG ?= configs/classification/default.yaml
CKPT ?=

train-cls:
	uv run python -m smoke_detection.cli.train --config configs/classification/default.yaml

train-seg:
	uv run python -m smoke_detection.cli.train --config configs/segmentation/default.yaml

eval-cls:
	uv run python -m smoke_detection.cli.eval --config configs/classification/default.yaml --ckpt $(CKPT)

eval-seg:
	uv run python -m smoke_detection.cli.eval --config configs/segmentation/default.yaml --ckpt $(CKPT)

clean:
	rm -rf lightning_logs/ .pytest_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
```

- [ ] **Step 2: Commit**

```bash
git add Makefile
git commit -m "chore: add Makefile shortcuts (install/lint/test/train/eval)"
```

---

### Task 5: Add GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

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
      - name: Pytest
        run: uv run pytest

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

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add lint/test/build GitHub Actions workflow"
```

---

### Task 6: Create scaffold directories and `tests/conftest.py`

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/.gitkeep`
- Create: `configs/.gitkeep`
- Create: `data/.gitkeep`
- Create: `notebooks/.gitkeep`

- [ ] **Step 1: Create `tests/conftest.py`**

```python
"""Pytest fixtures shared across the test suite.

Intentionally minimal for the initial scaffolding. Add fixtures here as tests
are introduced (e.g. a tiny synthetic dataset fixture for DataModule tests).
"""

import pytest


@pytest.fixture
def tiny_fake_dataset():
    """Placeholder fixture. Replace with a real synthetic dataset factory when
    test coverage is added."""
    return None
```

- [ ] **Step 2: Create empty `.gitkeep` files**

```bash
mkdir -p tests configs data notebooks
touch tests/.gitkeep configs/.gitkeep data/.gitkeep notebooks/.gitkeep
```

- [ ] **Step 3: Verify pytest runs cleanly**

```bash
uv run pytest
```

Expected: `no tests ran in 0.XXs` (exit code 5 is OK — no tests yet). If exit code is 5, wrap with a lenient check during CI later; for now it's fine because pytest returns 5 only on true no-collection and current matrix treats 0 and 5 as acceptable via our `addopts` — actually pytest's default exit is 5 on no-tests. Fix by creating a trivial placeholder test:

Create `tests/test_placeholder.py`:

```python
def test_package_imports():
    """Smoke-level: the package must be importable."""
    import smoke_detection

    assert smoke_detection.__version__
```

Re-run:

```bash
uv run pytest -v
```

Expected: `1 passed`.

- [ ] **Step 4: Commit**

```bash
git add tests/ configs/ data/ notebooks/
git commit -m "chore: add test scaffolding, empty configs/data/notebooks dirs"
```

---

## Phase 2 — Docs Reorganization

### Task 7: Move legacy docs to `docs/legacy/`

**Files:**
- Move + rename: `docs/MODEL_ARCHITECTURE.md` → `docs/legacy/MODEL_ARCHITECTURE_12ch.md`
- Move: `docs/MODEL_ARCHITECTURE_SLIDES_OUTLINE.md` → `docs/legacy/MODEL_ARCHITECTURE_SLIDES_OUTLINE.md`

- [ ] **Step 1: Create legacy dir and move files**

```bash
mkdir -p docs/legacy
git mv docs/MODEL_ARCHITECTURE.md docs/legacy/MODEL_ARCHITECTURE_12ch.md
git mv docs/MODEL_ARCHITECTURE_SLIDES_OUTLINE.md docs/legacy/MODEL_ARCHITECTURE_SLIDES_OUTLINE.md
```

- [ ] **Step 2: Prepend a legacy banner to `docs/legacy/MODEL_ARCHITECTURE_12ch.md`**

Add this as the new first two lines (before the existing `# Industrial Smoke Plume Detection - Model Architecture` heading):

```markdown
> **LEGACY DOCUMENT — preserved for reference.** This describes the original
> 12-channel (Sentinel-2 bands 1–10, 12, 13) model from the NeurIPS 2020
> paper. The current repository ships a 4-channel (B2, B3, B4, B8) pipeline.
> See `docs/MODEL_ARCHITECTURE.md` for the current architecture.

```

- [ ] **Step 3: Prepend a legacy banner to `docs/legacy/MODEL_ARCHITECTURE_SLIDES_OUTLINE.md`**

Add this as the new first two lines:

```markdown
> **LEGACY DOCUMENT — preserved for reference.** Slide outline for the
> original 12-channel model. Retained only for historical/reference use.

```

- [ ] **Step 4: Commit**

```bash
git add docs/
git commit -m "docs: move legacy 12-channel docs to docs/legacy/ with banner"
```

---

### Task 8: Write new stub `docs/MODEL_ARCHITECTURE.md`

**Files:**
- Create: `docs/MODEL_ARCHITECTURE.md`

- [ ] **Step 1: Write `docs/MODEL_ARCHITECTURE.md`**

```markdown
# Industrial Smoke Plume Detection — Model Architecture

This document describes the **current 4-channel** model architecture. The
original 12-channel design from the 2020 NeurIPS paper is archived under
`docs/legacy/`.

## Inputs

- Sentinel-2 GeoTIFF chips (`.tif`), normalized to `120 × 120` spatial size.
- **Four channels**, in this order: `B2` (blue, 490 nm), `B3` (green, 560 nm),
  `B4` (red, 665 nm), `B8` (NIR, 842 nm).
- Per-channel normalization with dataset-level mean/std (see
  `smoke_detection.data.transforms.Normalize`).

## Classification

- Backbone: `torchvision.models.resnet50` with ImageNet weights.
- First conv replaced to accept 4 input channels.
- Final FC replaced with `Linear(2048, 1)` producing a single logit.
- Loss: `BCEWithLogitsLoss`. Metric: accuracy, AUC.
- Training augmentation: random flips, 90-degree rotations, random crop to
  `90 × 90`.

## Segmentation

- Backbone: a 4-channel U-Net (from milesial/Pytorch-UNet, GPL v3).
- Output: per-pixel logit map; binarized at threshold `0`.
- Loss: `BCEWithLogitsLoss`. Metrics: IoU, image-level accuracy, area ratio.
- Same augmentation as classification, applied consistently to image + mask.

## Training framework

Both tasks are trained with **PyTorch Lightning**. `LightningDataModule`s own
the data pipeline; `LightningModule`s own loss/optimizer/metrics; the CLI
(`src/smoke_detection/cli/train.py`) constructs a `Trainer` with
`TensorBoardLogger` + `ModelCheckpoint` and wires everything from a YAML
config.

See `configs/classification/default.yaml` and `configs/segmentation/default.yaml`
for the full set of hyperparameters.
```

- [ ] **Step 2: Commit**

```bash
git add docs/MODEL_ARCHITECTURE.md
git commit -m "docs: add current 4-channel architecture overview"
```

---

## Phase 3 — Delete `deprecated/`

### Task 9: Remove `deprecated/` tree

**Files:**
- Delete: `deprecated/` (entire directory)

- [ ] **Step 1: Remove the directory**

```bash
git rm -r deprecated/
```

- [ ] **Step 2: Commit**

```bash
git commit -m "chore: remove legacy 12-channel code (preserved in git history)"
```

---

## Phase 4 — Package Internals Move

### Task 10: Create new package subdirectories with `__init__.py`

**Files:**
- Create: `src/smoke_detection/common/__init__.py`
- Create: `src/smoke_detection/data/__init__.py`
- Create: `src/smoke_detection/models/__init__.py`
- Create: `src/smoke_detection/training/__init__.py`
- Create: `src/smoke_detection/evaluation/__init__.py`
- Create: `src/smoke_detection/configs/__init__.py`
- Create: `src/smoke_detection/cli/__init__.py`

- [ ] **Step 1: Create the directories and empty `__init__.py` files**

```bash
cd src/smoke_detection
mkdir -p common data models training evaluation configs cli
touch common/__init__.py data/__init__.py models/__init__.py \
      training/__init__.py evaluation/__init__.py configs/__init__.py \
      cli/__init__.py
cd -
```

- [ ] **Step 2: Verify package still imports**

```bash
uv run python -c "import smoke_detection; print(smoke_detection.__version__)"
```

Expected: prints `0.2.0` (updated in Task 1).

- [ ] **Step 3: Bump `smoke_detection/__init__.py`**

Edit `src/smoke_detection/__init__.py`:

```python
__version__ = "0.2.0"
```

- [ ] **Step 4: Commit**

```bash
git add src/smoke_detection/
git commit -m "refactor: add empty subpackages (common/data/models/training/evaluation/configs/cli)"
```

---

### Task 11: Move `dataset_paths.py` → `common/paths.py`

**Files:**
- Create: `src/smoke_detection/common/paths.py`
- Delete: `src/smoke_detection/dataset_paths.py`

The default data root changes to the in-repo `data/` directory (per spec section 6).

- [ ] **Step 1: Write `src/smoke_detection/common/paths.py`**

```python
"""Filesystem paths for the prepared dataset.

Default layout (created by ``scripts/prepare_dataset.py``):

    <repo_root>/data/
        classification/{train,val,test}/{positive,negative}/*.tif
        segmentation/{train,val,test}/images/{positive,negative}/*.tif
        segmentation/{train,val,test}/labels/*.json

Override the root via ``SMOKEDET_DATA_ROOT`` env var or the ``paths.data_root``
YAML field.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/smoke_detection/common/paths.py -> src/smoke_detection/common ->
# src/smoke_detection -> src -> repo_root
_REPO_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT: Path = Path(
    os.environ.get("SMOKEDET_DATA_ROOT", str(_REPO_ROOT / "data"))
).resolve()


def classification_split(name: str, root: Path | None = None) -> Path:
    """Return the classification split directory for ``name`` in
    (``"train"``, ``"val"``, ``"test"``)."""
    base = (root or DATASET_ROOT) / "classification" / name
    return base


def segmentation_split(name: str, root: Path | None = None) -> tuple[Path, Path]:
    """Return ``(images_dir, labels_dir)`` for the segmentation split ``name``."""
    base = (root or DATASET_ROOT) / "segmentation" / name
    return base / "images", base / "labels"
```

- [ ] **Step 2: Delete the old module**

```bash
git rm src/smoke_detection/dataset_paths.py
```

- [ ] **Step 3: Update `scripts/prepare_dataset.py` default output**

In `scripts/prepare_dataset.py`, find this block (around lines 68–71):

```python
    p.add_argument(
        "--output",
        default=os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset_prepared")),
        help="Output root for prepared splits",
    )
```

Replace with:

```python
    p.add_argument(
        "--output",
        default=os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data")),
        help="Output root for prepared splits",
    )
```

And the bottom print line (around line 202):

```python
    print("Use dataset_paths.py (DATASET_ROOT) from the training scripts.")
```

Replace with:

```python
    print("Use smoke_detection.common.paths.DATASET_ROOT from training code.")
```

- [ ] **Step 4: Commit**

```bash
git add src/smoke_detection/common/paths.py src/smoke_detection/dataset_paths.py scripts/prepare_dataset.py
git commit -m "refactor: move dataset_paths -> common.paths, default data/ inside repo"
```

---

### Task 12: Add `common/seed.py` and `common/logging.py`

**Files:**
- Create: `src/smoke_detection/common/seed.py`
- Create: `src/smoke_detection/common/logging.py`

- [ ] **Step 1: Write `src/smoke_detection/common/seed.py`**

```python
"""Deterministic seeding for torch, numpy, and Python's ``random``."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed all RNGs and, optionally, enable deterministic cuDNN."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

- [ ] **Step 2: Write `src/smoke_detection/common/logging.py`**

```python
"""Minimal logging helper. Lightning handles its own stdout; this module is a
one-liner for CLI entry points that want a plain logger."""

from __future__ import annotations

import logging


def get_logger(name: str = "smoke_detection") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

- [ ] **Step 3: Commit**

```bash
git add src/smoke_detection/common/seed.py src/smoke_detection/common/logging.py
git commit -m "feat(common): add seed_everything and get_logger helpers"
```

---

### Task 13: Port classification dataset + shared transforms

**Files:**
- Create: `src/smoke_detection/data/transforms.py`
- Create: `src/smoke_detection/data/classification_dataset.py`
- Delete: `src/smoke_detection/classification/data.py`
- Delete: `src/smoke_detection/classification/__init__.py`

- [ ] **Step 1: Write `src/smoke_detection/data/transforms.py`**

Consolidates the duplicated `Normalize`/`Randomize`/`RandomCrop`/`ToTensor` classes from both tasks. The segmentation transforms also operate on the `fpt` mask when present.

```python
"""Shared transforms for classification and segmentation samples.

Samples are dicts: classification produces ``{'idx', 'img', 'lbl', 'imgfile'}``
and segmentation produces ``{'idx', 'img', 'fpt', 'imgfile'}``.
The transforms below inspect which keys are present and act accordingly.
"""

from __future__ import annotations

import numpy as np
import torch

# Dataset-level statistics for Sentinel-2 bands B2, B3, B4, B8.
CHANNEL_MEANS = np.array([900.5, 1061.4, 1091.7, 2186.3], dtype=np.float32)
CHANNEL_STDS = np.array([624.7, 640.8, 718.1, 947.9], dtype=np.float32)


class Normalize:
    """Per-channel normalization with fixed dataset statistics."""

    def __init__(self, means: np.ndarray = CHANNEL_MEANS, stds: np.ndarray = CHANNEL_STDS):
        self.means = means.reshape(-1, 1, 1)
        self.stds = stds.reshape(-1, 1, 1)

    def __call__(self, sample: dict) -> dict:
        sample["img"] = (sample["img"] - self.means) / self.stds
        return sample


class Randomize:
    """Random horizontal/vertical flips and 90-degree rotations.

    Applied consistently to the image and to the segmentation mask, if one is
    present in the sample under the ``fpt`` key.
    """

    def __call__(self, sample: dict) -> dict:
        img = sample["img"]
        fpt = sample.get("fpt")

        if np.random.randint(0, 2):
            img = np.flip(img, 2)
            if fpt is not None:
                fpt = np.flip(fpt, 1)
        if np.random.randint(0, 2):
            img = np.flip(img, 1)
            if fpt is not None:
                fpt = np.flip(fpt, 0)
        rot = np.random.randint(0, 4)
        img = np.rot90(img, rot, axes=(1, 2))
        if fpt is not None:
            fpt = np.rot90(fpt, rot, axes=(0, 1))

        sample["img"] = img.copy()
        if fpt is not None:
            sample["fpt"] = fpt.copy()
        return sample


class RandomCrop:
    """Random crop from 120x120 to a square of size ``crop``."""

    def __init__(self, crop: int = 90):
        self.crop = crop

    def __call__(self, sample: dict) -> dict:
        max_offset = 120 - self.crop
        x, y = np.random.randint(0, max_offset + 1, 2)
        sample["img"] = sample["img"][:, y : y + self.crop, x : x + self.crop].copy()
        if "fpt" in sample:
            sample["fpt"] = sample["fpt"][y : y + self.crop, x : x + self.crop].copy()
        return sample


class ToTensor:
    """Convert the ``img`` (and ``fpt`` if present) ndarray to ``torch.Tensor``."""

    def __call__(self, sample: dict) -> dict:
        sample["img"] = torch.from_numpy(np.ascontiguousarray(sample["img"])).float()
        if "fpt" in sample:
            sample["fpt"] = torch.from_numpy(np.ascontiguousarray(sample["fpt"])).float()
        return sample
```

- [ ] **Step 2: Write `src/smoke_detection/data/classification_dataset.py`**

Ports `SmokePlumeDataset` from the old file with minor cleanups: imports from `common.paths`, keeps the upsample/downsample balance logic, uses the new transforms module, and removes the module-level side-effect seeds.

```python
"""4-channel smoke plume classification Dataset.

Expects the layout produced by ``scripts/prepare_dataset.py``::

    <root>/classification/{train,val,test}/{positive,negative}/*.tif
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import rasterio as rio
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from smoke_detection.common.paths import classification_split
from smoke_detection.data.transforms import Normalize, RandomCrop, Randomize, ToTensor


class SmokePlumeDataset(Dataset):
    """Per-class-folder dataset for smoke plume classification.

    :param datadir: root containing ``positive`` and ``negative`` subdirs; if
        ``None``, defaults to ``classification_split('train')``.
    :param mult: multiply dataset length by this factor (for oversampling).
    :param transform: callable applied to each sample dict.
    :param balance: ``"upsample"``, ``"downsample"``, or anything else (no-op).
    """

    def __init__(
        self,
        datadir: str | Path | None = None,
        mult: int = 1,
        transform=None,
        balance: str = "upsample",
    ):
        if datadir is None:
            datadir = classification_split("train")
        self.datadir = Path(datadir)
        self.transform = transform

        imgfiles: list[str] = []
        labels: list[bool] = []
        positive_indices: list[int] = []
        negative_indices: list[int] = []

        idx = 0
        for root, _dirs, files in os.walk(self.datadir):
            for filename in files:
                if not filename.endswith(".tif"):
                    continue
                imgfiles.append(os.path.join(root, filename))
                if "positive" in root:
                    labels.append(True)
                    positive_indices.append(idx)
                    idx += 1
                elif "negative" in root:
                    labels.append(False)
                    negative_indices.append(idx)
                    idx += 1

        self.imgfiles = np.array(imgfiles)
        self.labels = np.array(labels)
        self.positive_indices = np.array(positive_indices)
        self.negative_indices = np.array(negative_indices)

        if balance == "downsample":
            self._balance_downsample()
        elif balance == "upsample":
            self._balance_upsample()

        if mult > 1:
            self.imgfiles = np.array([*self.imgfiles] * mult)
            self.labels = np.array([*self.labels] * mult)
            self.positive_indices = np.array([*self.positive_indices] * mult)
            self.negative_indices = np.array([*self.negative_indices] * mult)

    def __len__(self) -> int:
        return len(self.imgfiles)

    def _balance_downsample(self) -> None:
        idc = np.ravel(
            [
                self.positive_indices,
                self.negative_indices[
                    np.random.randint(
                        0, len(self.negative_indices), len(self.positive_indices)
                    )
                ],
            ]
        ).astype(int)
        self.imgfiles = self.imgfiles[idc]
        self.labels = self.labels[idc]
        self.positive_indices = np.arange(len(self.labels))[self.labels == True]  # noqa: E712
        self.negative_indices = np.arange(len(self.labels))[self.labels == False]  # noqa: E712

    def _balance_upsample(self) -> None:
        extra = np.random.randint(
            0,
            len(self.positive_indices),
            max(0, len(self.negative_indices) - len(self.positive_indices)),
        ).astype(int)
        extra_idc = self.positive_indices[extra]
        self.imgfiles = np.concatenate([self.imgfiles, self.imgfiles[extra_idc]])
        self.labels = np.concatenate([self.labels, self.labels[extra_idc]])
        self.positive_indices = np.arange(len(self.labels))[self.labels == True]  # noqa: E712
        self.negative_indices = np.arange(len(self.labels))[self.labels == False]  # noqa: E712

    def __getitem__(self, idx: int) -> dict:
        imgfile = rio.open(self.imgfiles[idx])
        # Sentinel-2 bands: B2(490), B3(560), B4(665), B8(842)
        imgdata = np.array([imgfile.read(i) for i in [2, 3, 4, 8]], dtype=np.float32)
        imgdata = _pad_to_120(imgdata)

        sample = {
            "idx": idx,
            "img": imgdata,
            "lbl": bool(self.labels[idx]),
            "imgfile": str(self.imgfiles[idx]),
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


def _pad_to_120(imgdata: np.ndarray) -> np.ndarray:
    """Right-/bottom-pad (by repeating the last row/col) to enforce 120x120."""
    if imgdata.shape[1] != 120:
        new = np.empty((imgdata.shape[0], 120, imgdata.shape[2]), dtype=imgdata.dtype)
        new[:, : imgdata.shape[1], :] = imgdata
        new[:, imgdata.shape[1] :, :] = imgdata[:, imgdata.shape[1] - 1 :, :]
        imgdata = new
    if imgdata.shape[2] != 120:
        new = np.empty((imgdata.shape[0], 120, 120), dtype=imgdata.dtype)
        new[:, :, : imgdata.shape[2]] = imgdata
        new[:, :, imgdata.shape[2] :] = imgdata[:, :, imgdata.shape[2] - 1 :]
        imgdata = new
    return imgdata


def build_default_transform(crop_size: int = 90) -> Compose:
    """Default training-time transform pipeline."""
    return Compose([Normalize(), RandomCrop(crop=crop_size), Randomize(), ToTensor()])


def build_eval_transform() -> Compose:
    """Eval-time transform pipeline (no random augmentation)."""
    return Compose([Normalize(), ToTensor()])
```

- [ ] **Step 3: Delete old files**

```bash
git rm src/smoke_detection/classification/data.py
```

(Keep `classification/__init__.py` for now — it's removed in Task 17.)

- [ ] **Step 4: Smoke check**

```bash
uv run python -c "from smoke_detection.data.classification_dataset import SmokePlumeDataset, build_default_transform; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add src/smoke_detection/data/ src/smoke_detection/classification/
git commit -m "refactor(data): port classification dataset + shared transforms"
```

---

### Task 14: Port classification model (pure `nn.Module`)

**Files:**
- Create: `src/smoke_detection/models/classifier_resnet.py`
- Delete: `src/smoke_detection/classification/model.py`

Remove the module-level `device` and singleton `model`; the new file exposes a `build_classifier()` function returning a fresh `nn.Module`.

- [ ] **Step 1: Write `src/smoke_detection/models/classifier_resnet.py`**

```python
"""Modified ResNet-50 classifier for 4-channel Sentinel-2 input."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def build_classifier(in_channels: int = 4, pretrained: bool = True) -> nn.Module:
    """Return a ResNet-50 with a 4-channel first conv and a single-logit head."""
    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet50(weights=weights)
    model.conv1 = nn.Conv2d(
        in_channels, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False
    )
    model.fc = nn.Linear(2048, 1)
    return model
```

- [ ] **Step 2: Delete the old file**

```bash
git rm src/smoke_detection/classification/model.py
```

- [ ] **Step 3: Smoke check**

```bash
uv run python -c "
import torch
from smoke_detection.models.classifier_resnet import build_classifier
m = build_classifier(in_channels=4, pretrained=False)
y = m(torch.randn(2, 4, 90, 90))
print('output shape:', tuple(y.shape))
"
```

Expected: prints `output shape: (2, 1)`.

- [ ] **Step 4: Commit**

```bash
git add src/smoke_detection/models/classifier_resnet.py src/smoke_detection/classification/model.py
git commit -m "refactor(models): port 4-channel ResNet-50 as pure nn.Module factory"
```

---

### Task 15: Port segmentation dataset

**Files:**
- Create: `src/smoke_detection/data/segmentation_dataset.py`
- Delete: `src/smoke_detection/segmentation/data.py`

Re-uses the new `transforms` module and `common.paths`. Preserves the Label-Studio JSON → rasterized mask pipeline.

- [ ] **Step 1: Write `src/smoke_detection/data/segmentation_dataset.py`**

```python
"""4-channel smoke plume segmentation Dataset.

Expects the layout produced by ``scripts/prepare_dataset.py``::

    <root>/segmentation/{train,val,test}/images/{positive,negative}/*.tif
    <root>/segmentation/{train,val,test}/labels/*.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.features import rasterize
from shapely.geometry import Polygon
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from smoke_detection.common.paths import segmentation_split
from smoke_detection.data.classification_dataset import _pad_to_120
from smoke_detection.data.transforms import Normalize, RandomCrop, Randomize, ToTensor


def label_image_url_to_tif_key(image_url: str) -> str:
    """Map a Label Studio image URL to the on-disk GeoTIFF basename."""
    key = "-".join(image_url.split("-")[1:]).replace(".png", ".tif")
    return key.replace(":", "_")


class SmokePlumeSegmentationDataset(Dataset):
    """Paired image + rasterized polygon-mask dataset."""

    def __init__(
        self,
        datadir: str | Path | None = None,
        seglabeldir: str | Path | None = None,
        mult: int = 1,
        transform=None,
    ):
        if datadir is None or seglabeldir is None:
            di, dl = segmentation_split("train")
            datadir = datadir if datadir is not None else di
            seglabeldir = seglabeldir if seglabeldir is not None else dl

        self.datadir = Path(datadir)
        self.transform = transform

        imgfiles: list[str] = []
        labels: list[bool] = []
        seglabels_per_img: list[list[np.ndarray]] = []
        positive_indices: list[int] = []
        negative_indices: list[int] = []

        raw_seglabels = []
        segfile_lookup: dict[str, int] = {}
        for i, seglabelfile in enumerate(os.listdir(seglabeldir)):
            with open(os.path.join(seglabeldir, seglabelfile)) as f:
                segdata = json.load(f)
            raw_seglabels.append(segdata)
            segfile_lookup[label_image_url_to_tif_key(segdata["data"]["image"])] = i

        idx = 0
        for root, _dirs, files in os.walk(self.datadir):
            for filename in files:
                if not filename.endswith(".tif"):
                    continue
                if filename not in segfile_lookup:
                    continue
                polygons: list[np.ndarray] = []
                for completions in raw_seglabels[segfile_lookup[filename]]["completions"]:
                    for result in completions["result"]:
                        pts = result["value"]["points"] + [result["value"]["points"][0]]
                        polygons.append(np.array(pts) * 1.2)
                if "positive" in root and polygons:
                    labels.append(True)
                    positive_indices.append(idx)
                    imgfiles.append(os.path.join(root, filename))
                    seglabels_per_img.append(polygons)
                    idx += 1

        # Pair each positive with an equal number of negatives (mirrors original code).
        n_pos = len(positive_indices)
        for root, _dirs, files in os.walk(self.datadir):
            for filename in files:
                if idx >= 2 * n_pos:
                    break
                if not filename.endswith(".tif"):
                    continue
                if "negative" in root:
                    labels.append(False)
                    negative_indices.append(idx)
                    imgfiles.append(os.path.join(root, filename))
                    seglabels_per_img.append([])
                    idx += 1

        self.imgfiles = np.array(imgfiles)
        self.labels = np.array(labels)
        self.positive_indices = np.array(positive_indices)
        self.negative_indices = np.array(negative_indices)
        self.seglabels = seglabels_per_img

        if mult > 1:
            self.imgfiles = np.array([*self.imgfiles] * mult)
            self.labels = np.array([*self.labels] * mult)
            self.positive_indices = np.array([*self.positive_indices] * mult)
            self.negative_indices = np.array([*self.negative_indices] * mult)
            self.seglabels = self.seglabels * mult

    def __len__(self) -> int:
        return len(self.imgfiles)

    def __getitem__(self, idx: int) -> dict:
        imgfile = rio.open(self.imgfiles[idx])
        imgdata = np.array([imgfile.read(i) for i in [2, 3, 4, 8]], dtype=np.float32)
        imgdata = _pad_to_120(imgdata)

        fptdata = np.zeros(imgdata.shape[1:], dtype=np.uint8)
        polygons = self.seglabels[idx]
        shapes: list[Polygon] = []
        if polygons:
            for pol in polygons:
                try:
                    shapes.append(Polygon(pol))
                except ValueError:
                    continue
            fptdata = rasterize(
                ((g, 1) for g in shapes), out_shape=fptdata.shape, all_touched=True
            )

        sample = {
            "idx": idx,
            "img": imgdata,
            "fpt": fptdata.astype(np.float32),
            "imgfile": str(self.imgfiles[idx]),
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


def build_default_transform(crop_size: int = 90) -> Compose:
    """Default training-time transform pipeline for segmentation."""
    return Compose([Normalize(), Randomize(), RandomCrop(crop=crop_size), ToTensor()])


def build_eval_transform() -> Compose:
    """Eval-time transform pipeline (no random augmentation)."""
    return Compose([Normalize(), ToTensor()])
```

- [ ] **Step 2: Delete the old file**

```bash
git rm src/smoke_detection/segmentation/data.py
```

- [ ] **Step 3: Smoke check**

```bash
uv run python -c "from smoke_detection.data.segmentation_dataset import SmokePlumeSegmentationDataset, build_default_transform; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add src/smoke_detection/data/segmentation_dataset.py src/smoke_detection/segmentation/data.py
git commit -m "refactor(data): port segmentation dataset + label loader"
```

---

### Task 16: Port U-Net segmenter model

**Files:**
- Create: `src/smoke_detection/models/segmenter_unet.py`
- Delete: `src/smoke_detection/segmentation/model.py`

- [ ] **Step 1: Write `src/smoke_detection/models/segmenter_unet.py`**

Copies the U-Net architecture from the old file verbatim (it was itself vendored from milesial/Pytorch-UNet, GPL v3 — retain that credit at the top). Removes the module-level `device`/`model` singletons and exposes a `build_segmenter()` factory.

```python
"""U-Net segmentation model for 4-channel Sentinel-2 input.

The U-Net code is adapted from milesial/Pytorch-UNet
(https://github.com/milesial/Pytorch-UNet), licensed GPL v3.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def build_segmenter(in_channels: int = 4, n_classes: int = 1, bilinear: bool = True) -> nn.Module:
    """Return a fresh 4-channel U-Net."""
    return UNet(n_channels=in_channels, n_classes=n_classes, bilinear=bilinear)
```

- [ ] **Step 2: Delete the old file**

```bash
git rm src/smoke_detection/segmentation/model.py
```

- [ ] **Step 3: Smoke check**

```bash
uv run python -c "
import torch
from smoke_detection.models.segmenter_unet import build_segmenter
m = build_segmenter()
y = m(torch.randn(2, 4, 96, 96))
print('output shape:', tuple(y.shape))
"
```

Expected: prints `output shape: (2, 1, 96, 96)`.

- [ ] **Step 4: Commit**

```bash
git add src/smoke_detection/models/segmenter_unet.py src/smoke_detection/segmentation/model.py
git commit -m "refactor(models): port 4-channel U-Net as pure nn.Module factory"
```

---

### Task 17: Remove old task-named subpackages

**Files:**
- Delete: `src/smoke_detection/classification/` (entire subdirectory)
- Delete: `src/smoke_detection/segmentation/` (entire subdirectory)

By this point, every file inside those directories has already been ported. Remaining contents are just the empty `__init__.py` and the old `train.py`/`eval.py`, which are superseded by `cli/train.py` and `cli/eval.py` built in Phase 6.

- [ ] **Step 1: Remove the directories**

```bash
git rm -r src/smoke_detection/classification/ src/smoke_detection/segmentation/
```

- [ ] **Step 2: Verify package still imports**

```bash
uv run python -c "
import smoke_detection
from smoke_detection.models.classifier_resnet import build_classifier
from smoke_detection.models.segmenter_unet import build_segmenter
from smoke_detection.data.classification_dataset import SmokePlumeDataset
from smoke_detection.data.segmentation_dataset import SmokePlumeSegmentationDataset
from smoke_detection.common.paths import DATASET_ROOT
from smoke_detection.common.seed import seed_everything
print('ok')
"
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor: remove obsolete classification/ and segmentation/ subpackages"
```

---

## Phase 5 — pydantic Configs + YAML

### Task 18: Write `configs/base.py` (shared pydantic schemas)

**Files:**
- Create: `src/smoke_detection/configs/base.py`

- [ ] **Step 1: Write `src/smoke_detection/configs/base.py`**

```python
"""Shared pydantic config schemas. Task-specific configs subclass ``BaseConfig``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_epochs: int = 50
    accelerator: Literal["auto", "cpu", "gpu"] = "auto"
    devices: int | str = "auto"
    precision: Literal["32", "16-mixed", "bf16-mixed"] = "32"
    deterministic: bool = True
    gradient_clip_val: float | None = None
    log_every_n_steps: int = 10
    fast_dev_run: bool = False


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_root: Path = Path("data")
    output_dir: Path = Path("lightning_logs")
    experiment_name: str


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lr: float
    momentum: float = 0.9
    weight_decay: float = 0.0
    scheduler: Literal["none", "cosine", "plateau"] = "none"


class BaseConfig(BaseSettings):
    """Top-level config. Subclasses add ``model:`` and ``data:`` blocks."""

    model_config = SettingsConfigDict(
        env_prefix="SMOKEDET_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    task: Literal["classification", "segmentation"]
    seed: int = 42
    trainer: TrainerConfig
    paths: PathsConfig
    optim: OptimConfig
```

- [ ] **Step 2: Commit**

```bash
git add src/smoke_detection/configs/base.py
git commit -m "feat(configs): add base pydantic schemas (trainer/paths/optim/base)"
```

---

### Task 19: Write task-specific config schemas

**Files:**
- Create: `src/smoke_detection/configs/classification.py`
- Create: `src/smoke_detection/configs/segmentation.py`

- [ ] **Step 1: Write `src/smoke_detection/configs/classification.py`**

```python
"""Classification experiment config."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from smoke_detection.configs.base import BaseConfig


class ClassificationModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backbone: Literal["resnet50"] = "resnet50"
    pretrained: bool = True
    in_channels: int = 4


class ClassificationDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = 32
    num_workers: int = 4
    crop_size: int = 90
    balance: Literal["upsample", "downsample", "none"] = "upsample"


class ClassificationConfig(BaseConfig):
    task: Literal["classification"] = "classification"
    model: ClassificationModelConfig
    data: ClassificationDataConfig
```

- [ ] **Step 2: Write `src/smoke_detection/configs/segmentation.py`**

```python
"""Segmentation experiment config."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from smoke_detection.configs.base import BaseConfig


class SegmentationModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    architecture: Literal["unet"] = "unet"
    in_channels: int = 4
    n_classes: int = 1
    bilinear: bool = True


class SegmentationDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = 16
    num_workers: int = 4
    crop_size: int = 90


class SegmentationConfig(BaseConfig):
    task: Literal["segmentation"] = "segmentation"
    model: SegmentationModelConfig
    data: SegmentationDataConfig
```

- [ ] **Step 3: Commit**

```bash
git add src/smoke_detection/configs/classification.py src/smoke_detection/configs/segmentation.py
git commit -m "feat(configs): add task-specific pydantic schemas (classification + segmentation)"
```

---

### Task 20: Write `configs/loader.py` (YAML + dotted overrides)

**Files:**
- Create: `src/smoke_detection/configs/loader.py`

- [ ] **Step 1: Write `src/smoke_detection/configs/loader.py`**

```python
"""YAML → typed config loader with simple dotted CLI overrides.

Precedence: YAML file < env vars (via pydantic-settings) < ``overrides`` list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from smoke_detection.configs.base import BaseConfig
from smoke_detection.configs.classification import ClassificationConfig
from smoke_detection.configs.segmentation import SegmentationConfig

_SCHEMAS: dict[str, type[BaseConfig]] = {
    "classification": ClassificationConfig,
    "segmentation": SegmentationConfig,
}


def load_config(path: str | Path, overrides: list[str] | None = None) -> BaseConfig:
    """Load and validate a YAML config; apply ``key=value`` dotted overrides."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if overrides:
        for item in overrides:
            _apply_dotted_override(raw, item)
    task = raw.get("task")
    if task not in _SCHEMAS:
        raise ValueError(
            f"Config {path} has invalid or missing 'task': {task!r}. "
            f"Expected one of {sorted(_SCHEMAS)}."
        )
    return _SCHEMAS[task].model_validate(raw)


def _apply_dotted_override(raw: dict[str, Any], override: str) -> None:
    """Apply one ``dotted.path=value`` mutation in place on ``raw``."""
    if "=" not in override:
        raise ValueError(f"Override must be in 'key=value' form, got: {override!r}")
    key, value = override.split("=", 1)
    keys = key.split(".")
    node = raw
    for part in keys[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot descend into non-mapping at key: {part}")
    node[keys[-1]] = _coerce_scalar(value)


def _coerce_scalar(value: str) -> Any:
    """Coerce CLI-style override strings to int/float/bool/null when unambiguous."""
    lower = value.lower()
    if lower in ("true", "false"):
        return lower == "true"
    if lower in ("null", "none"):
        return None
    try:
        if "." in value or "e" in lower:
            return float(value)
        return int(value)
    except ValueError:
        return value
```

- [ ] **Step 2: Smoke check**

```bash
uv run python -c "
from smoke_detection.configs.loader import _apply_dotted_override, _coerce_scalar
raw = {'optim': {'lr': 0.1}}
_apply_dotted_override(raw, 'optim.lr=1e-4')
print(raw)
assert raw == {'optim': {'lr': 1e-4}}
assert _coerce_scalar('true') is True
assert _coerce_scalar('42') == 42
assert _coerce_scalar('hello') == 'hello'
print('ok')
"
```

Expected: prints dict then `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/smoke_detection/configs/loader.py
git commit -m "feat(configs): add YAML loader with dotted overrides"
```

---

### Task 21: Write default YAML configs

**Files:**
- Create: `configs/classification/default.yaml`
- Create: `configs/segmentation/default.yaml`

- [ ] **Step 1: Write `configs/classification/default.yaml`**

```yaml
# Default classification experiment config.
# Override via:
#   SMOKEDET_OPTIM__LR=1e-3 python -m smoke_detection.cli.train --config configs/classification/default.yaml
# or:
#   python -m smoke_detection.cli.train --config configs/classification/default.yaml --override optim.lr=1e-3

task: classification
seed: 42
trainer:
  max_epochs: 50
  accelerator: auto
  devices: auto
  precision: "32"
  deterministic: true
  log_every_n_steps: 10
paths:
  data_root: data
  output_dir: lightning_logs
  experiment_name: classification_4ch_resnet50
optim:
  lr: 1.0e-4
  momentum: 0.9
  weight_decay: 0.0
  scheduler: plateau
model:
  backbone: resnet50
  pretrained: true
  in_channels: 4
data:
  batch_size: 32
  num_workers: 4
  crop_size: 90
  balance: upsample
```

- [ ] **Step 2: Write `configs/segmentation/default.yaml`**

```yaml
# Default segmentation experiment config.
task: segmentation
seed: 42
trainer:
  max_epochs: 300
  accelerator: auto
  devices: auto
  precision: "32"
  deterministic: true
  log_every_n_steps: 10
paths:
  data_root: data
  output_dir: lightning_logs
  experiment_name: segmentation_4ch_unet
optim:
  lr: 1.0e-4
  momentum: 0.9
  weight_decay: 0.0
  scheduler: plateau
model:
  architecture: unet
  in_channels: 4
  n_classes: 1
  bilinear: true
data:
  batch_size: 16
  num_workers: 4
  crop_size: 90
```

- [ ] **Step 3: Smoke check**

```bash
uv run python -c "
from smoke_detection.configs.loader import load_config
c = load_config('configs/classification/default.yaml')
print(c.task, c.optim.lr, c.model.backbone)
s = load_config('configs/segmentation/default.yaml')
print(s.task, s.model.architecture)
"
```

Expected: prints

```
classification 0.0001 resnet50
segmentation unet
```

- [ ] **Step 4: Commit**

```bash
git add configs/classification/ configs/segmentation/
git commit -m "feat(configs): add default YAML configs for classification and segmentation"
```

---

## Phase 6 — Port to Lightning

### Task 22: Write classification `LightningDataModule`

**Files:**
- Create: `src/smoke_detection/data/classification_datamodule.py`

- [ ] **Step 1: Write `src/smoke_detection/data/classification_datamodule.py`**

```python
"""LightningDataModule wrapping ``SmokePlumeDataset`` for classification."""

from __future__ import annotations

import platform
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader, RandomSampler

from smoke_detection.common.paths import classification_split
from smoke_detection.data.classification_dataset import (
    SmokePlumeDataset,
    build_default_transform,
    build_eval_transform,
)


class ClassificationDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root: Path,
        batch_size: int = 32,
        num_workers: int = 4,
        crop_size: int = 90,
        balance: str = "upsample",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_root = Path(data_root).resolve()
        self.batch_size = batch_size
        self.num_workers = 0 if platform.system() == "Windows" else num_workers
        self.crop_size = crop_size
        self.balance = balance

    def setup(self, stage: str | None = None) -> None:
        train_tfm = build_default_transform(crop_size=self.crop_size)
        eval_tfm = build_eval_transform()
        if stage in (None, "fit"):
            self.train_ds = SmokePlumeDataset(
                datadir=classification_split("train", self.data_root),
                transform=train_tfm,
                balance=self.balance,
            )
            self.val_ds = SmokePlumeDataset(
                datadir=classification_split("val", self.data_root),
                transform=eval_tfm,
                balance="none",
            )
        if stage in (None, "test", "predict"):
            self.test_ds = SmokePlumeDataset(
                datadir=classification_split("test", self.data_root),
                transform=eval_tfm,
                balance="none",
            )

    def train_dataloader(self) -> DataLoader:
        sampler = RandomSampler(
            self.train_ds, replacement=True, num_samples=max(1, 2 * len(self.train_ds) // 3)
        )
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
```

- [ ] **Step 2: Commit**

```bash
git add src/smoke_detection/data/classification_datamodule.py
git commit -m "feat(data): add ClassificationDataModule"
```

---

### Task 23: Write segmentation `LightningDataModule`

**Files:**
- Create: `src/smoke_detection/data/segmentation_datamodule.py`

- [ ] **Step 1: Write `src/smoke_detection/data/segmentation_datamodule.py`**

```python
"""LightningDataModule wrapping ``SmokePlumeSegmentationDataset``."""

from __future__ import annotations

import platform
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader, RandomSampler

from smoke_detection.common.paths import segmentation_split
from smoke_detection.data.segmentation_dataset import (
    SmokePlumeSegmentationDataset,
    build_default_transform,
    build_eval_transform,
)


class SegmentationDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root: Path,
        batch_size: int = 16,
        num_workers: int = 4,
        crop_size: int = 90,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_root = Path(data_root).resolve()
        self.batch_size = batch_size
        self.num_workers = 0 if platform.system() == "Windows" else num_workers
        self.crop_size = crop_size

    def setup(self, stage: str | None = None) -> None:
        train_tfm = build_default_transform(crop_size=self.crop_size)
        eval_tfm = build_eval_transform()
        if stage in (None, "fit"):
            tr_img, tr_lbl = segmentation_split("train", self.data_root)
            va_img, va_lbl = segmentation_split("val", self.data_root)
            self.train_ds = SmokePlumeSegmentationDataset(
                datadir=tr_img, seglabeldir=tr_lbl, transform=train_tfm
            )
            self.val_ds = SmokePlumeSegmentationDataset(
                datadir=va_img, seglabeldir=va_lbl, transform=eval_tfm
            )
        if stage in (None, "test", "predict"):
            te_img, te_lbl = segmentation_split("test", self.data_root)
            self.test_ds = SmokePlumeSegmentationDataset(
                datadir=te_img, seglabeldir=te_lbl, transform=eval_tfm
            )

    def train_dataloader(self) -> DataLoader:
        sampler = RandomSampler(
            self.train_ds, replacement=True, num_samples=max(1, 2 * len(self.train_ds) // 3)
        )
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
```

- [ ] **Step 2: Commit**

```bash
git add src/smoke_detection/data/segmentation_datamodule.py
git commit -m "feat(data): add SegmentationDataModule"
```

---

### Task 24: Write classification `LightningModule`

**Files:**
- Create: `src/smoke_detection/training/classification_module.py`

- [ ] **Step 1: Write `src/smoke_detection/training/classification_module.py`**

```python
"""LightningModule for 4-channel smoke plume classification."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn
from torch import optim
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

from smoke_detection.models.classifier_resnet import build_classifier


class ClassificationModule(L.LightningModule):
    def __init__(
        self,
        in_channels: int = 4,
        pretrained: bool = True,
        lr: float = 1e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        scheduler: str = "none",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = build_classifier(in_channels=in_channels, pretrained=pretrained)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.val_auc = BinaryAUROC()
        self.test_acc = BinaryAccuracy()
        self.test_auc = BinaryAUROC()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _shared_step(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch["img"]
        y = batch["lbl"].float().reshape(-1, 1)
        logits = self.net(x)
        loss = self.loss_fn(logits, y)
        return loss, logits, y

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, logits, y = self._shared_step(batch)
        self.train_acc.update(torch.sigmoid(logits).squeeze(1), y.squeeze(1).int())
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, logits, y = self._shared_step(batch)
        probs = torch.sigmoid(logits).squeeze(1)
        self.val_acc.update(probs, y.squeeze(1).int())
        self.val_auc.update(probs, y.squeeze(1).int())
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        _, logits, y = self._shared_step(batch)
        probs = torch.sigmoid(logits).squeeze(1)
        self.test_acc.update(probs, y.squeeze(1).int())
        self.test_auc.update(probs, y.squeeze(1).int())
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt = optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.scheduler == "plateau":
            sched = optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, threshold=1e-4, min_lr=1e-6
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val/loss"}}
        if self.hparams.scheduler == "cosine":
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
            return {"optimizer": opt, "lr_scheduler": sched}
        return opt
```

- [ ] **Step 2: Commit**

```bash
git add src/smoke_detection/training/classification_module.py
git commit -m "feat(training): add ClassificationModule (Lightning)"
```

---

### Task 25: Write segmentation `LightningModule`

**Files:**
- Create: `src/smoke_detection/training/segmentation_module.py`

- [ ] **Step 1: Write `src/smoke_detection/training/segmentation_module.py`**

```python
"""LightningModule for 4-channel smoke plume segmentation."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn
from torch import optim
from torchmetrics.classification import BinaryAccuracy, BinaryJaccardIndex

from smoke_detection.models.segmenter_unet import build_segmenter


class SegmentationModule(L.LightningModule):
    def __init__(
        self,
        in_channels: int = 4,
        n_classes: int = 1,
        bilinear: bool = True,
        lr: float = 1e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        scheduler: str = "none",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = build_segmenter(
            in_channels=in_channels, n_classes=n_classes, bilinear=bilinear
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_iou = BinaryJaccardIndex()
        self.val_iou = BinaryJaccardIndex()
        self.val_acc = BinaryAccuracy()
        self.test_iou = BinaryJaccardIndex()
        self.test_acc = BinaryAccuracy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _shared_step(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch["img"]
        y = batch["fpt"].float().unsqueeze(1)  # [B, 1, H, W]
        logits = self.net(x)
        loss = self.loss_fn(logits, y)
        return loss, logits, y

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, logits, y = self._shared_step(batch)
        preds = (logits >= 0).int()
        self.train_iou.update(preds, y.int())
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/iou", self.train_iou, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, logits, y = self._shared_step(batch)
        preds = (logits >= 0).int()
        self.val_iou.update(preds, y.int())
        # Image-level presence (per image: any predicted mask vs. any true mask).
        image_pred = (preds.sum(dim=(1, 2, 3)) > 0).int()
        image_true = (y.sum(dim=(1, 2, 3)) > 0).int()
        self.val_acc.update(image_pred, image_true)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/iou", self.val_iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/img_acc", self.val_acc, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        _, logits, y = self._shared_step(batch)
        preds = (logits >= 0).int()
        self.test_iou.update(preds, y.int())
        image_pred = (preds.sum(dim=(1, 2, 3)) > 0).int()
        image_true = (y.sum(dim=(1, 2, 3)) > 0).int()
        self.test_acc.update(image_pred, image_true)
        self.log("test/iou", self.test_iou, on_step=False, on_epoch=True)
        self.log("test/img_acc", self.test_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt = optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.scheduler == "plateau":
            sched = optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, threshold=1e-4, min_lr=1e-6
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val/loss"}}
        if self.hparams.scheduler == "cosine":
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
            return {"optimizer": opt, "lr_scheduler": sched}
        return opt
```

- [ ] **Step 2: Commit**

```bash
git add src/smoke_detection/training/segmentation_module.py
git commit -m "feat(training): add SegmentationModule (Lightning)"
```

---

### Task 26: Write evaluation plotting helpers

**Files:**
- Create: `src/smoke_detection/evaluation/classification_metrics.py`
- Create: `src/smoke_detection/evaluation/segmentation_metrics.py`

- [ ] **Step 1: Write `src/smoke_detection/evaluation/classification_metrics.py`**

Preserves the confusion-matrix and ROC plotting from the old `classification/eval.py`.

```python
"""Classification eval plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def plot_confusion_matrix(
    tp: int, tn: int, fp: int, fn: int, out_path: str | Path
) -> None:
    """Save a 2x2 confusion-matrix heatmap."""
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred Neg", "Pred Pos"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True Neg", "True Pos"])
    ax.set_title("Confusion Matrix")
    for r in range(2):
        for c in range(2):
            ax.text(
                c, r, str(cm[r, c]),
                ha="center", va="center", fontsize=14,
                color="white" if cm[r, c] > cm.max() / 2 else "black",
            )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(scores: list[float], labels: list[int], out_path: str | Path) -> None:
    """Save an ROC-curve plot; annotates AUC."""
    fpr, tpr, _ = roc_curve(labels, scores)
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float("nan")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 2: Write `src/smoke_detection/evaluation/segmentation_metrics.py`**

```python
"""Segmentation eval plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_iou_distribution(ious: list[float], out_path: str | Path) -> None:
    """Save an IoU distribution histogram."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ious, bins=20, edgecolor="black")
    mean = float(np.mean(ious)) if ious else 0.0
    ax.axvline(mean, color="red", linestyle="--", label=f"Mean = {mean:.3f}")
    ax.set_xlabel("IoU Score")
    ax.set_ylabel("Count")
    ax.set_title(f"IoU Distribution (n={len(ious)})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_area_ratio_distribution(ratios: list[float], out_path: str | Path) -> None:
    """Save a pred/true area-ratio histogram."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ratios, bins=30, edgecolor="black")
    ax.axvline(1.0, color="red", linestyle="--", label="Reference = 1.0")
    mean = float(np.mean(ratios)) if ratios else 0.0
    ax.axvline(mean, color="orange", linestyle="--", label=f"Mean = {mean:.3f}")
    ax.set_xlabel("Area Ratio (pred / true)")
    ax.set_ylabel("Count")
    ax.set_title(f"Area Ratio Distribution (n={len(ratios)})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 3: Commit**

```bash
git add src/smoke_detection/evaluation/
git commit -m "feat(evaluation): add plotting helpers for classification and segmentation"
```

---

### Task 27: Write `cli/train.py`

**Files:**
- Create: `src/smoke_detection/cli/train.py`

- [ ] **Step 1: Write `src/smoke_detection/cli/train.py`**

```python
"""CLI entry point for training. Dispatches on ``config.task``."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from smoke_detection.common.logging import get_logger
from smoke_detection.common.seed import seed_everything
from smoke_detection.configs.base import BaseConfig
from smoke_detection.configs.classification import ClassificationConfig
from smoke_detection.configs.loader import load_config
from smoke_detection.configs.segmentation import SegmentationConfig
from smoke_detection.data.classification_datamodule import ClassificationDataModule
from smoke_detection.data.segmentation_datamodule import SegmentationDataModule
from smoke_detection.training.classification_module import ClassificationModule
from smoke_detection.training.segmentation_module import SegmentationModule

log = get_logger(__name__)


def _build_classification(cfg: ClassificationConfig):
    module = ClassificationModule(
        in_channels=cfg.model.in_channels,
        pretrained=cfg.model.pretrained,
        lr=cfg.optim.lr,
        momentum=cfg.optim.momentum,
        weight_decay=cfg.optim.weight_decay,
        scheduler=cfg.optim.scheduler,
    )
    datamodule = ClassificationDataModule(
        data_root=cfg.paths.data_root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        crop_size=cfg.data.crop_size,
        balance=cfg.data.balance,
    )
    monitor = "val/loss"
    return module, datamodule, monitor


def _build_segmentation(cfg: SegmentationConfig):
    module = SegmentationModule(
        in_channels=cfg.model.in_channels,
        n_classes=cfg.model.n_classes,
        bilinear=cfg.model.bilinear,
        lr=cfg.optim.lr,
        momentum=cfg.optim.momentum,
        weight_decay=cfg.optim.weight_decay,
        scheduler=cfg.optim.scheduler,
    )
    datamodule = SegmentationDataModule(
        data_root=cfg.paths.data_root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        crop_size=cfg.data.crop_size,
    )
    monitor = "val/loss"
    return module, datamodule, monitor


def _build_trainer(cfg: BaseConfig, monitor: str) -> L.Trainer:
    logger = TensorBoardLogger(
        save_dir=str(cfg.paths.output_dir),
        name=cfg.paths.experiment_name,
    )
    checkpoint_cb = ModelCheckpoint(
        monitor=monitor,
        mode="min",
        save_top_k=3,
        save_last=True,
        filename="epoch{epoch:03d}-{val/loss:.4f}",
        auto_insert_metric_name=False,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    return L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        deterministic=cfg.trainer.deterministic,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        fast_dev_run=cfg.trainer.fast_dev_run,
        logger=logger,
        callbacks=[checkpoint_cb, lr_cb],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a smoke-detection model")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted override, e.g. --override optim.lr=1e-4 (repeatable)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.override)
    seed_everything(cfg.seed, deterministic=cfg.trainer.deterministic)
    log.info("Loaded config: task=%s experiment=%s", cfg.task, cfg.paths.experiment_name)

    if isinstance(cfg, ClassificationConfig):
        module, datamodule, monitor = _build_classification(cfg)
    elif isinstance(cfg, SegmentationConfig):
        module, datamodule, monitor = _build_segmentation(cfg)
    else:
        raise RuntimeError(f"Unsupported config type: {type(cfg).__name__}")

    trainer = _build_trainer(cfg, monitor=monitor)
    trainer.fit(module, datamodule=datamodule)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke check (import + fast_dev_run wiring, no data)**

```bash
uv run python -c "
from smoke_detection.cli.train import _build_trainer
from smoke_detection.configs.loader import load_config
cfg = load_config('configs/classification/default.yaml', overrides=['trainer.fast_dev_run=true'])
t = _build_trainer(cfg, monitor='val/loss')
print('trainer built:', type(t).__name__, 'fast_dev_run=', cfg.trainer.fast_dev_run)
"
```

Expected: prints `trainer built: Trainer fast_dev_run= True`.

- [ ] **Step 3: Commit**

```bash
git add src/smoke_detection/cli/train.py
git commit -m "feat(cli): add unified train entry point dispatching on config.task"
```

---

### Task 28: Write `cli/eval.py`

**Files:**
- Create: `src/smoke_detection/cli/eval.py`

Runs `trainer.test()` from a checkpoint and writes plotting artifacts into the experiment output directory.

- [ ] **Step 1: Write `src/smoke_detection/cli/eval.py`**

```python
"""CLI entry point for evaluation. Runs ``trainer.test`` and dumps plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from lightning import Trainer

from smoke_detection.common.logging import get_logger
from smoke_detection.common.seed import seed_everything
from smoke_detection.configs.classification import ClassificationConfig
from smoke_detection.configs.loader import load_config
from smoke_detection.configs.segmentation import SegmentationConfig
from smoke_detection.data.classification_datamodule import ClassificationDataModule
from smoke_detection.data.segmentation_datamodule import SegmentationDataModule
from smoke_detection.evaluation.classification_metrics import (
    plot_confusion_matrix,
    plot_roc_curve,
)
from smoke_detection.evaluation.segmentation_metrics import (
    plot_area_ratio_distribution,
    plot_iou_distribution,
)
from smoke_detection.training.classification_module import ClassificationModule
from smoke_detection.training.segmentation_module import SegmentationModule

log = get_logger(__name__)


def _eval_classification(cfg: ClassificationConfig, ckpt: Path, out_dir: Path) -> None:
    dm = ClassificationDataModule(
        data_root=cfg.paths.data_root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        crop_size=cfg.data.crop_size,
        balance="none",
    )
    module = ClassificationModule.load_from_checkpoint(str(ckpt))
    trainer = Trainer(accelerator=cfg.trainer.accelerator, devices=cfg.trainer.devices)
    trainer.test(module, datamodule=dm)

    dm.setup(stage="test")
    module.eval()
    scores: list[float] = []
    labels: list[int] = []
    tp = tn = fp = fn = 0
    with torch.no_grad():
        for batch in dm.test_dataloader():
            logits = module(batch["img"].to(module.device)).cpu().squeeze(1)
            probs = torch.sigmoid(logits).tolist()
            ys = batch["lbl"].int().tolist()
            scores.extend(probs)
            labels.extend(ys)
            for p, y in zip(probs, ys):
                pred = 1 if p >= 0.5 else 0
                if pred == 1 and y == 1:
                    tp += 1
                elif pred == 0 and y == 0:
                    tn += 1
                elif pred == 1 and y == 0:
                    fp += 1
                else:
                    fn += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(tp, tn, fp, fn, out_dir / "confusion_matrix.png")
    plot_roc_curve(scores, labels, out_dir / "roc_curve.png")
    log.info("Wrote classification eval plots to %s", out_dir)


def _eval_segmentation(cfg: SegmentationConfig, ckpt: Path, out_dir: Path) -> None:
    dm = SegmentationDataModule(
        data_root=cfg.paths.data_root,
        batch_size=1,
        num_workers=cfg.data.num_workers,
        crop_size=cfg.data.crop_size,
    )
    module = SegmentationModule.load_from_checkpoint(str(ckpt))
    trainer = Trainer(accelerator=cfg.trainer.accelerator, devices=cfg.trainer.devices)
    trainer.test(module, datamodule=dm)

    dm.setup(stage="test")
    module.eval()
    ious: list[float] = []
    ratios: list[float] = []
    with torch.no_grad():
        for batch in dm.test_dataloader():
            y = batch["fpt"].float().unsqueeze(1).to(module.device)
            logits = module(batch["img"].to(module.device))
            preds = (logits >= 0).float()
            inter = (preds * y).sum(dim=(1, 2, 3))
            union = ((preds + y) > 0).float().sum(dim=(1, 2, 3))
            for k in range(y.shape[0]):
                if y[k].sum() > 0 and preds[k].sum() > 0:
                    ious.append(float(inter[k] / union[k]))
                a_pred = float(preds[k].sum())
                a_true = float(y[k].sum())
                if a_pred == 0 and a_true == 0:
                    ratios.append(1.0)
                elif a_true == 0:
                    ratios.append(0.0)
                else:
                    ratios.append(a_pred / a_true)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_iou_distribution(ious, out_dir / "iou_distribution.png")
    plot_area_ratio_distribution(ratios, out_dir / "area_ratio_distribution.png")
    log.info(
        "Segmentation eval: mean IoU=%.4f mean area ratio=%.4f (n=%d)",
        float(np.mean(ious)) if ious else 0.0,
        float(np.mean(ratios)) if ratios else 0.0,
        len(ious),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained smoke-detection model")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument(
        "--override", action="append", default=[], help="Dotted overrides (repeatable)"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for eval plots (defaults to <output_dir>/<experiment_name>/eval).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.override)
    seed_everything(cfg.seed, deterministic=cfg.trainer.deterministic)

    out_dir = args.out_dir or (cfg.paths.output_dir / cfg.paths.experiment_name / "eval")
    if isinstance(cfg, ClassificationConfig):
        _eval_classification(cfg, args.ckpt, out_dir)
    elif isinstance(cfg, SegmentationConfig):
        _eval_segmentation(cfg, args.ckpt, out_dir)
    else:
        raise RuntimeError(f"Unsupported config type: {type(cfg).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke check (import only)**

```bash
uv run python -c "from smoke_detection.cli import eval as _; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/smoke_detection/cli/eval.py
git commit -m "feat(cli): add unified eval entry point with plotting artifacts"
```

---

### Task 29: End-to-end Lightning `fast_dev_run` smoke test on synthetic data

**Files:**
- Create: `scripts/smoketest_fast_dev_run.py`

Generates a tiny synthetic prepared-dataset tree under `/tmp/smokedet_smoke/` (or `$TEMP/smokedet_smoke/` on Windows), points `SMOKEDET_DATA_ROOT` at it, and runs `trainer.fit` with `fast_dev_run=True` for both tasks. Serves as the acceptance-criteria smoke check.

- [ ] **Step 1: Write `scripts/smoketest_fast_dev_run.py`**

```python
"""End-to-end smoke test: run Lightning fast_dev_run on a synthetic dataset.

Creates a tiny prepared-dataset tree, points SMOKEDET_DATA_ROOT at it, and
runs training for one batch on both the classification and segmentation
configs. Exits non-zero on any exception.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.transform import from_origin


def _write_fake_tif(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (np.random.rand(13, 120, 120) * 3000).astype(np.uint16)
    profile = {
        "driver": "GTiff",
        "width": 120,
        "height": 120,
        "count": 13,
        "dtype": "uint16",
        "transform": from_origin(0, 0, 1, 1),
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(data)


def _write_fake_label(path: Path, tif_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completions": [
            {
                "result": [
                    {"value": {"points": [[10, 10], [60, 10], [60, 60], [10, 60]]}}
                ]
            }
        ],
        "data": {"image": f"/data/upload/1-{tif_name.replace('.tif', '.png')}"},
    }
    path.write_text(json.dumps(payload))


def _build_fake_dataset(root: Path) -> None:
    for split in ("train", "val", "test"):
        for cls in ("positive", "negative"):
            for i in range(4):
                fname = f"{cls}_{split}_{i:03d}.tif"
                _write_fake_tif(root / "classification" / split / cls / fname)
                _write_fake_tif(root / "segmentation" / split / "images" / cls / fname)
                if cls == "positive":
                    _write_fake_label(
                        root / "segmentation" / split / "labels" / f"{fname}.json", fname
                    )


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="smokedet_smoke_"))
    try:
        _build_fake_dataset(tmp)
        os.environ["SMOKEDET_DATA_ROOT"] = str(tmp)
        from smoke_detection.cli.train import main as train_main

        print(f"[smoketest] data root: {tmp}")
        print("[smoketest] running classification fast_dev_run…")
        rc = train_main(
            [
                "--config",
                "configs/classification/default.yaml",
                "--override",
                "trainer.fast_dev_run=true",
                "--override",
                "data.batch_size=2",
                "--override",
                "data.num_workers=0",
            ]
        )
        if rc != 0:
            return rc

        print("[smoketest] running segmentation fast_dev_run…")
        rc = train_main(
            [
                "--config",
                "configs/segmentation/default.yaml",
                "--override",
                "trainer.fast_dev_run=true",
                "--override",
                "data.batch_size=2",
                "--override",
                "data.num_workers=0",
            ]
        )
        return rc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the smoke test**

```bash
uv run python scripts/smoketest_fast_dev_run.py
```

Expected: completes without exception and exits 0. Prints `fast_dev_run=True` progress from Lightning for both tasks.

If it fails: fix the issue rather than skipping. Common issues to look for:
- `fast_dev_run` string-to-bool coercion in `loader._coerce_scalar` — it already handles `"true"` → `True`.
- Windows `num_workers` — forced to 0 by `--override data.num_workers=0`.
- Missing `lightning_logs/` permission — the script writes into the repo's `lightning_logs/` by default; make sure it's writable.

- [ ] **Step 3: Commit**

```bash
git add scripts/smoketest_fast_dev_run.py
git commit -m "test: add end-to-end Lightning fast_dev_run smoketest on synthetic data"
```

---

## Phase 7 — README, CONTRIBUTING, CHANGELOG

### Task 30: Rewrite `README.md`

**Files:**
- Modify: `README.md` (complete rewrite)

- [ ] **Step 1: Overwrite `README.md`**

```markdown
# Industrial Smoke Plume Detection

This repository implements a two-stage PyTorch Lightning pipeline for detecting
and segmenting industrial smoke plumes from Sentinel-2 multispectral satellite
imagery (4 channels: B2, B3, B4, B8).

Based on the publication *Characterization of Industrial Smoke Plumes from
Remote Sensing Data*, NeurIPS 2020 *Tackling Climate Change with Machine
Learning* workshop.

![segmentation example images](assets/segmentation.png "Segmentation Example Images")

## Project Structure

```
configs/                 experiment YAMLs
data/                    prepared dataset (gitignored)
docs/                    architecture + legacy docs
notebooks/               exploratory notebooks
scripts/                 dataset preparation + smoke tests
src/smoke_detection/
  common/                seed, paths, logging
  data/                  Datasets + LightningDataModules + transforms
  models/                pure nn.Module definitions (ResNet-50, U-Net)
  training/              LightningModules (loss, optimizer, metrics, steps)
  evaluation/            plotting helpers (confusion matrix, ROC, IoU hist)
  configs/               pydantic schemas + YAML loader
  cli/                   train.py, eval.py entry points
tests/                   pytest scaffolding
```

## Installation

Requires Python >= 3.11 (3.12 recommended; see `.python-version`) and
[`uv`](https://github.com/astral-sh/uv).

    uv sync --extra dev

Or from pip/venv:

    pip install -e ".[dev]"

## Data Preparation

1. Download the dataset from [Zenodo](http://doi.org/10.5281/zenodo.4250706)
   and extract it (you should have a `4250706/` directory containing
   `images/` and `segmentation_labels/`).
2. Generate train/val/test splits into the default `data/` location:

```bash
python scripts/prepare_dataset.py --source /path/to/4250706 --output data
```

Override the default root via `SMOKEDET_DATA_ROOT=/some/path`.

The expected layout after preparation:

```
data/
  classification/{train,val,test}/{positive,negative}/*.tif
  segmentation/{train,val,test}/images/{positive,negative}/*.tif
  segmentation/{train,val,test}/labels/*.json
```

## Training

All training goes through a single CLI, parameterized by YAML config:

```bash
# Classification
python -m smoke_detection.cli.train --config configs/classification/default.yaml

# Segmentation
python -m smoke_detection.cli.train --config configs/segmentation/default.yaml

# Override any config field via dotted overrides
python -m smoke_detection.cli.train \
    --config configs/classification/default.yaml \
    --override optim.lr=1e-3 \
    --override trainer.max_epochs=10
```

Or via `make`:

```bash
make train-cls        # classification
make train-seg        # segmentation
```

Logs, checkpoints, and TensorBoard events land in
`lightning_logs/<experiment_name>/version_N/`.

## Evaluation

```bash
python -m smoke_detection.cli.eval \
    --config configs/segmentation/default.yaml \
    --ckpt lightning_logs/segmentation_4ch_unet/version_0/checkpoints/last.ckpt
```

The eval CLI writes confusion matrix + ROC plots (classification) or IoU +
area-ratio histograms (segmentation) into
`lightning_logs/<experiment_name>/eval/`.

## Dev Loop

```bash
make install          # uv sync --extra dev
make lint             # ruff check + ruff format --check
make format           # auto-fix
make test             # pytest
make clean
```

Pre-commit hooks:

```bash
uv run pre-commit install
```

## Citation

    Mommert, M., Sigel, M., Neuhausler, M., Scheibenreif, L., Borth, D.,
    "Characterization of Industrial Smoke Plumes from Remote Sensing Data",
    Tackling Climate Change with Machine Learning Workshop, NeurIPS 2020.

## License

GPL v3 — see `LICENSE`. U-Net code adapted from
[milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) (GPL v3).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for Lightning-based pipeline + config-driven CLI"
```

---

### Task 31: Write `CONTRIBUTING.md`

**Files:**
- Create: `CONTRIBUTING.md`

- [ ] **Step 1: Write `CONTRIBUTING.md`**

```markdown
# Contributing

Thanks for your interest in contributing.

## Dev Setup

1. Install [`uv`](https://github.com/astral-sh/uv).
2. Clone the repo and install dev dependencies:

   ```bash
   uv sync --extra dev
   ```

3. Install pre-commit hooks:

   ```bash
   uv run pre-commit install
   ```

## Branching

- Branch from `master`: `git checkout -b feat/<short-name>` or `fix/<short-name>`.
- Open a PR against `master`.
- CI (lint + test + build) must pass before merging.

## Code Style

- Formatter + linter: [`ruff`](https://docs.astral.sh/ruff/) (config in `pyproject.toml`).
- Run locally:

  ```bash
  make lint     # check
  make format   # auto-fix
  ```

- Line length 100. Target Python 3.11.

## Tests

- Use `pytest`. Tests live under `tests/`.
- Run: `make test` or `uv run pytest`.
- The test suite currently contains scaffolding only. Add real tests as
  behavior is stabilized.

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) prefixes:

- `feat(scope): …` – new functionality
- `fix(scope): …` – bug fix
- `refactor(scope): …` – no behavior change
- `docs: …`, `chore: …`, `ci: …`, `test: …`

## Running Experiments

- Experiment configs live in `configs/<task>/*.yaml`.
- Never hardcode hyperparameters in Python code — add a field to the pydantic
  schema in `src/smoke_detection/configs/` and reference it from the
  LightningModule / DataModule.
- `lightning_logs/` is gitignored. Do not commit checkpoints or TensorBoard
  event files.
```

- [ ] **Step 2: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "docs: add CONTRIBUTING guide"
```

---

### Task 32: Write `CHANGELOG.md`

**Files:**
- Create: `CHANGELOG.md`

- [ ] **Step 1: Write `CHANGELOG.md`**

```markdown
# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] — 2026-04-17

### Added
- PyTorch Lightning training/eval (`LightningModule`, `LightningDataModule`).
- pydantic v2 + YAML experiment configs under `configs/`.
- Unified CLI: `python -m smoke_detection.cli.train --config <yaml>` and
  `smoke_detection.cli.eval`.
- `ruff` (lint + format), `pytest` scaffolding, `pre-commit`, GitHub Actions
  CI (lint / test matrix 3.11 & 3.12 / build).
- `Makefile` shortcuts.
- `.python-version` pin (3.12 for dev).
- `docs/legacy/` preserving the original 12-channel architecture notes.

### Changed
- Reorganized `src/smoke_detection/` into
  `common/ data/ models/ training/ evaluation/ configs/ cli/`.
- Default data root is now the in-repo `data/` directory (was a sibling
  `dataset_prepared/`). Override via `SMOKEDET_DATA_ROOT` or
  `paths.data_root` YAML field.
- Dependencies consolidated into `pyproject.toml`.
- Python floor raised to 3.11.
- `pyproject.toml` `build-backend` fixed (`setuptools.build_meta`).

### Removed
- `deprecated/` (original 12-channel code). Retained in git history.
- `requirements.txt` (deps now in `pyproject.toml`).
- Task-named subpackages `src/smoke_detection/classification/` and
  `src/smoke_detection/segmentation/`.
- `src/smoke_detection/dataset_paths.py` (replaced by `common/paths.py`).

### Breaking
- Previous entry points `python -m smoke_detection.classification.train`
  and `.segmentation.train` no longer exist. Use
  `python -m smoke_detection.cli.train --config <yaml>`.
- Old `.model` state-dict files from the 12-channel era are not loadable
  into the new 4-channel Lightning modules.
- Metric reporting now uses `torchmetrics`; numeric values may differ
  slightly from the previous hand-rolled aggregation (per-image vs.
  batch-averaged). Paper-accuracy parity has not been re-verified post-port.

## [0.1.0] — Initial release

- Classification (ResNet-50) and segmentation (U-Net) for Sentinel-2 smoke
  plume detection, argparse-based scripts.
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: add CHANGELOG for 0.2.0 cleanup release"
```

---

## Verification Checklist

After the last task, verify each acceptance criterion from
`docs/superpowers/specs/2026-04-17-repo-cleanup-design.md` §10:

- [ ] `uv sync --extra dev` succeeds on 3.11 and 3.12 (CI matrix does this).
- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass clean.
- [ ] `uv run pytest` exits 0 (one placeholder test passes).
- [ ] `uv run python scripts/smoketest_fast_dev_run.py` completes without
  exception (covers both `cli.train` configs with `fast_dev_run=True`).
- [ ] CI workflow passes on a PR.
- [ ] `deprecated/` is gone; `docs/legacy/` contains the two legacy docs.
- [ ] README walks a new reader end-to-end without referencing removed paths.

If anything fails, fix it before declaring done.
