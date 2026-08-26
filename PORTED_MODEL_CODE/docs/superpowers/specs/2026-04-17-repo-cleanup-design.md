# Repo Cleanup — Standard Software + ML Layout

**Status:** Approved design
**Date:** 2026-04-17
**Scope:** Restructure the Industrial Smoke Plume Detection repo to match a standard Python software project layout and a standard ML project layout, port training/evaluation to PyTorch Lightning, and introduce config-driven experiments.

---

## 1. Goals and Non-Goals

### Goals
- Active-development repo: CI, linting, pre-commit, tests scaffolding.
- Standard ML layout: clear separation of `data/`, `models/`, `training/`, `evaluation/`, `configs/`, `cli/`.
- Port training/eval from hand-rolled loops to **PyTorch Lightning** (user is learning Lightning).
- Config-driven experiments via **pydantic (v2) + YAML**, with env-var and CLI overrides.
- Delete legacy 12-channel code; preserve its *documentation* under `docs/legacy/`.
- Consolidate deps in `pyproject.toml` (drop `requirements.txt`); fix broken build backend.

### Non-Goals (explicit YAGNI)
- Writing actual tests beyond scaffolding.
- MLflow/Weights & Biases integration (TensorBoard via Lightning is sufficient).
- Hydra / OmegaConf (pydantic is the chosen config stack).
- Docker / devcontainer / `mypy` / coverage gates.
- Multi-GPU or distributed configs beyond Lightning defaults.
- Preserving the old `python -m smoke_detection.classification.train` entry-point shape (breaking change — documented in CHANGELOG).

---

## 2. Target Repository Layout

```
.
├── .github/workflows/ci.yml            # ruff + pytest + build
├── .pre-commit-config.yaml
├── .python-version                     # 3.12
├── .gitignore                          # expanded
├── CHANGELOG.md                        # Keep-a-Changelog
├── CONTRIBUTING.md
├── LICENSE                             # unchanged
├── Makefile
├── README.md                           # rewritten
├── pyproject.toml                      # fixed backend, all deps here
├── uv.lock
├── assets/segmentation.png
├── configs/
│   ├── classification/default.yaml
│   └── segmentation/default.yaml
├── data/                               # gitignored, default dataset root
│   └── .gitkeep
├── docs/
│   ├── MODEL_ARCHITECTURE.md           # current (4-channel) system
│   ├── legacy/
│   │   ├── MODEL_ARCHITECTURE_12ch.md
│   │   └── MODEL_ARCHITECTURE_SLIDES_OUTLINE.md
│   └── superpowers/specs/…
├── notebooks/
│   └── .gitkeep
├── scripts/
│   └── prepare_dataset.py
├── src/smoke_detection/
│   ├── __init__.py
│   ├── common/ { paths.py, seed.py, logging.py }
│   ├── data/
│   │   ├── transforms.py
│   │   ├── classification_dataset.py
│   │   ├── segmentation_dataset.py
│   │   ├── classification_datamodule.py
│   │   └── segmentation_datamodule.py
│   ├── models/ { classifier_resnet.py, segmenter_unet.py }
│   ├── training/ { classification_module.py, segmentation_module.py }
│   ├── evaluation/ { classification_metrics.py, segmentation_metrics.py }
│   ├── configs/ { base.py, classification.py, segmentation.py, loader.py }
│   └── cli/ { train.py, eval.py }
└── tests/
    ├── conftest.py
    └── .gitkeep
```

Removed:
- `deprecated/` (entire tree).
- `requirements.txt` (deps live in `pyproject.toml`).
- `src/smoke_detection/dataset_paths.py` (replaced by `common/paths.py`).
- `src/smoke_detection/classification/{data,model,train,eval}.py` and
  `src/smoke_detection/segmentation/{data,model,train,eval}.py`
  (code moves into the new subpackages; the old task-named subpackages are removed).

---

## 3. Package Architecture

### 3.1 Layer responsibilities

| Layer | Owns | Does NOT own |
|---|---|---|
| `models/` | Pure `nn.Module` definitions. Framework-agnostic (testable standalone). | Loss, optimizer, metrics, training loops. |
| `data/` | `Dataset` classes, `LightningDataModule`s, transforms. | Model code, training loops. |
| `training/` | `LightningModule` per task — owns loss, optimizer, scheduler, `training_step` / `validation_step` / `test_step`, logged metrics. | Raw `nn.Module` construction (imports from `models/`). |
| `evaluation/` | Metric computation helpers + plotting (confusion matrix, ROC, IoU distribution, area-ratio distribution) invoked by `cli/eval.py` and by LightningModule hooks where useful. | Training concerns. |
| `configs/` | pydantic schemas + YAML loader. | Hyperparameter values (those live in `configs/` YAML files at repo root). |
| `cli/` | Thin entry points. Parse `--config` + `--override`, build the LightningModule + DataModule + Trainer, call `fit()` / `test()`. | Business logic. |
| `common/` | Seed helper, path helpers, logging setup. | Anything task-specific. |

### 3.2 CLI shape

```bash
python -m smoke_detection.cli.train --config configs/classification/default.yaml
python -m smoke_detection.cli.train --config configs/segmentation/default.yaml --override optim.lr=1e-4

python -m smoke_detection.cli.eval \
    --config configs/segmentation/default.yaml \
    --ckpt lightning_logs/seg_4ch_unet/version_3/checkpoints/best.ckpt
```

The CLI dispatches on `config.task` (`"classification"` or `"segmentation"`) to choose the right LightningModule + DataModule.

### 3.3 Lightning responsibilities

- **`LightningModule`** owns: loss, optimizer, scheduler, train/val/test step, logged metrics (`torchmetrics`), `configure_optimizers`.
- **`LightningDataModule`** owns: `setup()` (builds train/val/test `Dataset`s), `train_dataloader` / `val_dataloader` / `test_dataloader`, transform composition.
- **`Trainer`** wiring (in `cli/train.py`): `TensorBoardLogger` (default), `ModelCheckpoint` (save best by val metric), `LearningRateMonitor`, `deterministic=True` by default.
- Checkpoints land in `{paths.output_dir}/{paths.experiment_name}/version_N/checkpoints/`.

---

## 4. Config System (pydantic + YAML)

### 4.1 Schema layers

`src/smoke_detection/configs/base.py`:

```python
class TrainerConfig(BaseModel):
    max_epochs: int = 50
    accelerator: Literal["auto", "cpu", "gpu"] = "auto"
    devices: int | str = "auto"
    precision: Literal["32", "16-mixed", "bf16-mixed"] = "32"
    deterministic: bool = True
    gradient_clip_val: float | None = None

class PathsConfig(BaseModel):
    data_root: Path = Path("data")          # resolved vs. repo root
    output_dir: Path = Path("lightning_logs")
    experiment_name: str

class OptimConfig(BaseModel):
    lr: float
    momentum: float = 0.9
    weight_decay: float = 0.0
    scheduler: Literal["none", "cosine", "step"] = "none"

class BaseConfig(BaseSettings):
    task: Literal["classification", "segmentation"]
    seed: int = 42
    trainer: TrainerConfig
    paths: PathsConfig
    optim: OptimConfig
    model_config = SettingsConfigDict(
        env_prefix="SMOKEDET_",
        env_nested_delimiter="__",
    )
```

Task-specific classes (`ClassificationConfig`, `SegmentationConfig`) subclass `BaseConfig` and add `model:` and `data:` sub-blocks.

### 4.2 Loader

```python
def load_config(path: Path, overrides: list[str] | None = None) -> BaseConfig:
    raw = yaml.safe_load(path.read_text())
    if overrides:
        raw = apply_dotted_overrides(raw, overrides)  # e.g. "optim.lr=1e-4"
    task = raw.get("task")
    schema = {
        "classification": ClassificationConfig,
        "segmentation":   SegmentationConfig,
    }[task]
    return schema.model_validate(raw)
```

### 4.3 Precedence
YAML defaults → environment variables (`SMOKEDET_OPTIM__LR=1e-4`) → CLI `--override` flags. Env-var support is free via `BaseSettings`.

### 4.4 Example YAML

```yaml
# configs/classification/default.yaml
task: classification
seed: 42
trainer:
  max_epochs: 50
  accelerator: auto
paths:
  experiment_name: classification_4ch_resnet50
optim:
  lr: 1.0e-4
  momentum: 0.9
model:
  backbone: resnet50
  pretrained: true
  in_channels: 4
data:
  batch_size: 32
  num_workers: 4
  crop_size: 90
```

---

## 5. Tooling

### 5.1 Python and dependencies
- **Floor:** `requires-python = ">=3.11"`.
- **Dev pin:** `.python-version` = `3.12`.
- **Build backend:** `setuptools.build_meta` (fixes current broken `setuptools.backends.legacy:build`).
- **Runtime deps (all in `pyproject.toml`):** `numpy`, `matplotlib`, `rasterio`, `scikit-learn`, `tensorboard`, `torch>=2.2`, `torchvision`, `tqdm`, `shapely`, `lightning>=2.2`, `torchmetrics>=1.3`, `pydantic>=2.6`, `pydantic-settings>=2.2`, `pyyaml>=6.0`.
- **Optional `[dev]`:** `ruff`, `pytest`, `pytest-cov`, `pre-commit`.
- **Optional `[notebooks]`:** `jupyter`, `ipykernel`.
- **Primary installer:** `uv`. `uv sync` reads `pyproject.toml` + `uv.lock`.
- **`requirements.txt`:** deleted.

### 5.2 Lint / format — `ruff`
`[tool.ruff]` in `pyproject.toml`: line-length 100, target `py311`, rules `E`, `F`, `W`, `I`, `B`, `UP`, `SIM`. `ruff format` replaces black.

### 5.3 Tests — `pytest`
`[tool.pytest.ini_options]`: `testpaths = ["tests"]`, `addopts = "-ra --strict-markers"`. `tests/conftest.py` defined but empty aside from a `tiny_fake_dataset` stub for future fixtures. No actual tests yet.

### 5.4 Pre-commit — `.pre-commit-config.yaml`
Hooks: `ruff-format`, `ruff` (check + autofix), `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-added-large-files`.

### 5.5 CI — `.github/workflows/ci.yml`
Triggers: push + PR to `master`.
- **lint** job: `uv sync --extra dev`, `ruff check`, `ruff format --check`.
- **test** job: matrix Python `3.11`, `3.12` on `ubuntu-latest`; `uv sync --extra dev`; `pytest`.
- **build** job: `uv build` (wheel + sdist sanity).
- CPU-only. No GPU runners.

### 5.6 Makefile
```
make install        # uv sync --extra dev
make lint           # ruff check + ruff format --check
make format         # ruff format
make test           # pytest
make train-cls CONFIG=configs/classification/default.yaml
make train-seg CONFIG=configs/segmentation/default.yaml
make clean          # remove lightning_logs, __pycache__, .pytest_cache
```

---

## 6. Data, Models, and Artifacts Conventions

- **Default data root:** `data/` at repo root (gitignored). Override with `SMOKEDET_DATA_ROOT` env var or `paths.data_root` in YAML.
- **Expected layout** (produced by `scripts/prepare_dataset.py`):
  ```
  data/
    classification/{train,val,test}/{smoke,no_smoke}/*.tif
    segmentation/{train,val,test}/{images,labels}/*.tif
  ```
- **Lightning logs / checkpoints:** `lightning_logs/<experiment_name>/version_N/` (gitignored).
- **Notebooks:** `notebooks/` (gitignored outputs, source notebooks checked in).

---

## 7. Docs Reorganization

- `docs/MODEL_ARCHITECTURE.md` — **rewritten** to describe the *current* 4-channel system. Initial version is a concise stub pointing out that (a) classification is a 4-channel ResNet-50 head-swap, (b) segmentation is a 4-channel U-Net, (c) detailed 12-channel history is in `docs/legacy/`. Deeper documentation can be fleshed out in a follow-up.
- `docs/legacy/MODEL_ARCHITECTURE_12ch.md` — renamed from the current `docs/MODEL_ARCHITECTURE.md` (it describes the old 12-channel system).
- `docs/legacy/MODEL_ARCHITECTURE_SLIDES_OUTLINE.md` — moved as-is.

---

## 8. Migration Sequence

Each phase is one commit; the repo stays importable at every phase boundary.

1. **Scaffolding & hygiene** — fix `pyproject.toml` backend, add all new runtime + dev deps, add `ruff`/`pytest`/`pre-commit` config blocks, add `Makefile`, add `.github/workflows/ci.yml`, add `.python-version`, expand `.gitignore` (lightning_logs, mlruns, ipynb_checkpoints, data/, notebooks/outputs), create empty `tests/`, `configs/`, `data/`, `notebooks/`. Delete `requirements.txt`.
2. **Docs reorg** — move old architecture docs to `docs/legacy/` (rename MD file to `_12ch`), write new stub `docs/MODEL_ARCHITECTURE.md`.
3. **Delete `deprecated/`** — one commit.
4. **Package internals move** — relocate code into `src/smoke_detection/{common,data,models,training,evaluation,configs,cli}/`. Rename `dataset_paths.py` → `common/paths.py`. Old task subpackages (`classification/`, `segmentation/` inside `src/smoke_detection/`) are removed in this phase.
5. **Introduce pydantic configs + YAML** — add config schemas, loader, default YAMLs. Add `cli/train.py` and `cli/eval.py`. (At this phase boundary, training runs through new entry points but may still use ported-but-not-yet-Lightning code.)
6. **Port to Lightning** — wrap models in `LightningModule`s, datasets in `LightningDataModule`s, replace hand-rolled loops with `Trainer.fit()` / `Trainer.test()`. Swap hand-rolled metrics to `torchmetrics` where equivalents exist; keep custom plotting helpers for ROC, IoU distribution, area-ratio distribution.
7. **README + CONTRIBUTING + CHANGELOG** — new usage instructions (`make`, `--config`), dev setup, first CHANGELOG entry describing the breaking refactor.

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Can't run real training during cleanup (no GPU/data here). | Import-level sanity checks on every phase; `Trainer(fast_dev_run=True)` on a synthetic 2-sample tensor batch during phase 6. Full correctness validation is the user's post-merge step. |
| `torchmetrics` IoU may not match the hand-rolled metric exactly (per-image vs. micro-averaged). | Keep the custom IoU helper in `evaluation/` alongside the torchmetrics version. CHANGELOG notes the parity question as a follow-up to verify against paper numbers. |
| Old `.model` checkpoints from the 12-channel code are unloadable into the new 4-channel Lightning modules. | Documented as a breaking change. `.gitignore` already excludes `*.model`. |
| External users following the old README invoke `python -m smoke_detection.classification.train`, which no longer exists. | Breaking change called out explicitly in README + CHANGELOG. New invocation (`cli.train --config ...`) documented prominently. |
| Lightning default logger directory differs from prior convention. | `paths.output_dir` in config controls it; default is `lightning_logs/` (gitignored). |

---

## 10. Acceptance Criteria

The cleanup is complete when:

1. `uv sync --extra dev` succeeds on Python 3.11 and 3.12.
2. `ruff check` and `ruff format --check` pass clean.
3. `pytest` runs and exits zero (no tests, but scaffolding valid).
4. `python -m smoke_detection.cli.train --config configs/classification/default.yaml` starts a Lightning training run up through the first batch under `Trainer(fast_dev_run=True)` on synthetic tensors (or on real data if present).
5. `python -m smoke_detection.cli.train --config configs/segmentation/default.yaml` same as above.
6. CI passes on a test PR.
7. `deprecated/` is gone; `docs/legacy/` holds the two preserved legacy docs.
8. README walks a new reader from `uv sync` → `prepare_dataset.py` → `cli.train` without referencing any removed path.

---

## 11. Out of Scope (explicit deferrals)

- Writing actual test cases.
- MLflow / Weights & Biases.
- Docker / devcontainer.
- `mypy` / coverage thresholds.
- Distributed / multi-GPU configs.
- Fleshing `docs/MODEL_ARCHITECTURE.md` beyond the initial stub.
- Verifying paper-accuracy reproduction after the Lightning port.
