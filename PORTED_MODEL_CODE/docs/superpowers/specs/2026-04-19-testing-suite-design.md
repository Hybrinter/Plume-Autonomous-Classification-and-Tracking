# Testing Suite — Design

**Date:** 2026-04-19
**Scope:** Introduce a unit/integration/e2e test suite for the `smoke_detection`
package, with behavior-oriented coverage, CI wiring, and a marker-based tiering
scheme. No tests for third-party libraries.

## 1. Goals and Non-Goals

### Goals

- Ship a pytest suite that covers every authored module under
  `src/smoke_detection/` and `scripts/` to **≥ 80% branch-and-line coverage**.
- Exercise the behavioral invariants that make the pipeline correct (Sentinel-2
  channel selection, image/mask augmentation synchronization, U-Net
  size-preservation, normalization statistics, loss-decrease sanity).
- Structure tests into three tiers selectable via pytest markers so CI stays
  fast and local dev can opt into heavier runs.
- Replace the ad-hoc `scripts/smoketest_fast_dev_run.py` with a first-class
  pytest e2e case.

### Non-Goals

- No benchmark or wall-clock assertions.
- No tests against the real Zenodo dataset; synthetic data only.
- No visual assertions on matplotlib output; file emission + PNG magic bytes
  only.
- No bit-for-bit CPU↔GPU parity; cuDNN is non-deterministic by design.
- Not chasing 100% coverage; CLI `if __name__ == "__main__"` tails excluded.

## 2. Architecture

### 2.1 Directory layout

Tests mirror `src/smoke_detection/` one-to-one under `tests/unit/`, with
separate `tests/integration/` and `tests/e2e/` folders for multi-component
tests.

```
tests/
  conftest.py                 # top-level: determinism, repo root, cached weights
  _data.py                    # synthetic dataset builders (module, not a fixture)
  unit/
    conftest.py
    common/
      test_paths.py
      test_seed.py
      test_logging.py
    configs/
      test_schemas.py
      test_loader.py
    data/
      test_transforms.py
      test_pad_to_120.py
      test_classification_dataset.py
      test_segmentation_dataset.py
    models/
      test_classifier_resnet.py
      test_segmenter_unet.py
    training/
      test_classification_module.py
      test_segmentation_module.py
    evaluation/
      test_classification_metrics.py
      test_segmentation_metrics.py
    cli/
      test_train_argparse.py
      test_eval_argparse.py
    scripts/
      test_prepare_dataset_helpers.py
  integration/
    conftest.py
    test_classification_datamodule.py
    test_segmentation_datamodule.py
    test_module_consumes_batch.py
    test_config_to_trainer.py
    test_prepare_dataset_script.py
  e2e/
    conftest.py
    test_fast_dev_run.py
    test_train_eval_cycle.py
    test_checkpoint_roundtrip.py
```

### 2.2 Tiers via pytest markers

| Marker  | Purpose                                                              |
|---------|----------------------------------------------------------------------|
| _none_  | Fast tier. Default. Unit + cheap integration. Target < 60s on CPU.   |
| `slow`  | Pretrained-weight tests, overfitting sanity loops (>~2s each).       |
| `gpu`   | Requires CUDA; auto-skipped when `torch.cuda.is_available()` False.  |
| `e2e`   | fast_dev_run, CLI end-to-end, checkpoint round-trip.                 |

Default invocation runs only the fast tier. Other tiers unlock via
`-m <marker>` or Makefile targets.

### 2.3 Make targets

```make
test:          # fast tier + slow tier (coverage-gated)
	uv run pytest -m "not slow and not e2e and not gpu"
	uv run pytest -m "slow" --cov-append --cov-fail-under=80
test-e2e:
	uv run pytest -m "e2e" --timeout=300
test-gpu:
	uv run pytest -m "gpu"
test-all:
	uv run pytest -m "not gpu"
```

## 3. Fixtures

### 3.1 Top-level (`tests/conftest.py`)

- **`_deterministic` (autouse, function-scoped).** Calls
  `smoke_detection.common.seed.seed_everything(1234, deterministic=True)`
  before every test so any use of `numpy`/`torch`/`random` is reproducible.
- **`repo_root` (session).** Absolute path to the repo root, resolved from
  `__file__`.
- **`synthetic_dataset_root` (session).** Builds a read-only prepared-dataset
  tree under `tmp_path_factory` with 13-band uint16 GeoTIFFs (120×120) and
  Label-Studio-style JSON masks. Populates both classification and
  segmentation subtrees, with enough sites per split that each is non-empty.
  Sets `SMOKEDET_DATA_ROOT` for the session. Lives at top level because it
  is consumed by unit (`data/`), integration, and e2e tests alike.
- **`mutable_dataset_root` (function).** Fresh copy of the synthetic tree
  per test. Used only where a test writes into it.
- **`tiny_classification_batch`, `tiny_segmentation_batch` (function).**
  Hand-built tensor dicts (`img`, `lbl`/`fpt`) for unit tests that need a
  real-looking batch without a DataModule.
- **`cached_resnet50_weights` (session).** Verifies that the
  `ResNet50_Weights.IMAGENET1K_V1` checkpoint is present in the Torch Hub
  cache (`TORCH_HOME`); downloads once if missing. Returns the weights enum.
  All `slow` tests that need pretrained weights depend on this fixture.
- **GPU skip hook.** Top-level conftest registers a `pytest_collection_modifyitems`
  hook that auto-skips `gpu`-marked items when CUDA is unavailable.
- **Lightning quiet.** `os.environ["PYTORCH_LIGHTNING_LOG_LEVEL"] = "ERROR"`
  is set on session start.

### 3.2 Integration & E2E scoped fixtures

Defined in `tests/integration/conftest.py` or `tests/e2e/conftest.py` as
appropriate:

- **`sample_classification_config`, `sample_segmentation_config` (function).**
  Pydantic-validated config objects with `fast_dev_run=true`, `num_workers=0`,
  `batch_size=2`, `data_root=synthetic_dataset_root`.
- **`classification_yaml_tmp`, `segmentation_yaml_tmp` (function).** Write
  the sample configs to temp YAML files and yield paths. Used by CLI e2e.

### 3.3 Synthetic dataset builders (`tests/_data.py`)

Two module-level helpers (not fixtures):

- **`build_synthetic_zenodo_source(root, n_positive=6, n_negative=6, n_sites=4)`** —
  produces `<root>/images/{positive,negative}/*.tif` and
  `<root>/segmentation_labels/*.json` in the shape `prepare_dataset.py`
  consumes. Used by `test_prepare_dataset_script.py`.
- **`build_synthetic_prepared_tree(root, n_positive=6, n_negative=6, n_sites=4)`** —
  produces the post-`prepare_dataset.py` layout directly. Used by
  `synthetic_dataset_root`.

Both use `numpy.random.default_rng(seed=<fixed>)` for reproducibility, emit
real 13-band rasterio GeoTIFFs (so `[2, 3, 4, 8]` indexing works), and write
label JSON containing a `completions[0].result[].value.points` polygon array
plus a `data.image` URL formatted like Label Studio's upload paths.

### 3.4 Cross-platform rules

- `num_workers` forced to 0 on Windows is covered by a monkeypatched
  `platform.system()` test.
- `prepare_dataset.py` `symlink`/`hardlink` modes are `skipif(sys.platform == "win32")`.
  `copy` mode runs everywhere.
- All paths are `pathlib.Path`.
- Teardown uses `shutil.rmtree(..., ignore_errors=True)` to tolerate Windows
  rasterio file-handle races.

## 4. Per-Module Test Matrix

Tests are labelled **B** (coverage test — exercises a branch/statement) or
**C** (behavioral-correctness test — asserts a scientific/semantic invariant).

### 4.1 `common/`

**`test_paths.py`**
- B: `classification_split` returns `<root>/classification/<split>` for every split.
- B: `segmentation_split` returns a `(images, labels)` tuple with the right suffixes.
- C: `SMOKEDET_DATA_ROOT` env var overrides `DATASET_ROOT` (via module reload inside a patched env).
- B: passing explicit `root=` kwarg overrides the module-level default.

**`test_seed.py`**
- C: same seed → identical `torch.rand`, `np.random.rand`, `random.random` sequences.
- C: different seeds → different sequences (guards against a no-op implementation).
- B: `deterministic=True` sets `torch.backends.cudnn.deterministic=True`, `benchmark=False`.
- B: `PYTHONHASHSEED` env var is set to the seed.

**`test_logging.py`**
- B: `get_logger(name)` returns a logger with exactly one `StreamHandler` even when called twice.
- B: handler format includes `%(asctime)s %(levelname)s %(name)s`.

### 4.2 `configs/`

**`test_schemas.py`**
- B: `TrainerConfig` defaults; rejects unknown field; rejects bad
  `precision`/`accelerator` literal.
- B: `PathsConfig` requires `experiment_name`.
- B: `OptimConfig` requires `lr`; validates `scheduler` literal.
- B: `ClassificationConfig` requires `model` + `data` blocks; `task` tag is `"classification"`.
- B: `SegmentationConfig` analogous with `task="segmentation"`.
- C: loading a classification YAML via `SegmentationConfig` raises a `ValidationError`.

**`test_loader.py`**
- B: `load_config` round-trips both bundled YAML defaults.
- B: `--override optim.lr=1e-3` mutates the raw dict before validation.
- B: nested override creates missing dicts (`trainer.fast_dev_run=true`).
- B: bad override syntax (no `=`) raises `ValueError`.
- B: override descending into a scalar raises `ValueError`.
- C: `_coerce_scalar` cases — `"true"/"false"` → bool; `"null"/"none"` → None;
  `"1.5"` → float; `"42"` → int; `"1e-4"` → float; unparseable → str.
- B: missing or invalid `task` key raises.

### 4.3 `data/`

**`test_transforms.py`**
- B: `Normalize` — hand-computed `(x - mean)/std` matches.
- C: `Normalize` — an input of all `CHANNEL_MEANS` yields zeros.
- C: stats broadcast correctly over `(C, H, W)`.
- B: `RandomCrop` produces shape `(C, crop, crop)` for seeded offsets.
- C: when `fpt` is present, `RandomCrop` uses the same offsets for image and mask.
- B: `Randomize` with seeded RNG is deterministic.
- C: `Randomize` flip matches between `img` and `fpt` — verified with a
  hand-crafted mask whose positive pixels are in a known corner.
- B: `ToTensor` returns `torch.float32`, contiguous tensors for `img` and `fpt`.

**`test_pad_to_120.py`**
- B: `_pad_to_120` on `(C, 120, 120)` is identity.
- C: `(C, 100, 120)` right-pads by repeating the last column.
- C: `(C, 120, 100)` bottom-pads by repeating the last row.
- C: `(C, 80, 100)` both dimensions → 120×120.

**`test_classification_dataset.py`** (uses `synthetic_dataset_root`)
- B: `SmokePlumeDataset.__len__` > 0.
- B: `__getitem__` returns dict with keys `{idx, img, lbl, imgfile}`.
- C: `img` has shape `(4, 120, 120)` dtype `float32` pre-transform.
- C: labels derived from path containing `positive`/`negative`.
- B: `balance="upsample"` equalizes classes; `"downsample"` reduces length;
  `"none"` leaves counts intact.
- B: `mult > 1` scales length.
- B: `transform` is applied when provided.
- C: upsample/downsample never drops the only positive or negative.

**`test_segmentation_dataset.py`** (uses `synthetic_dataset_root`)
- B: `label_image_url_to_tif_key` strips prefix, replaces `:` with `_`, `.png`→`.tif`.
- B: `__len__` equals 2× positive count (paired-negatives rule).
- B: `__getitem__` yields `{idx, img, fpt, imgfile}`; `fpt` is `(120, 120)` float32.
- C: rasterized mask non-zero for positives, all-zero for negatives.
- C: polygons scaled by 1.2 — verify via a hand-crafted polygon whose
  rasterized area matches expectation ± 10%.
- B: malformed polygon (<3 unique points) is skipped without raising.

### 4.4 `models/`

**`test_classifier_resnet.py`**
- B: `build_classifier(pretrained=False)` returns an `nn.Module`.
- C: `conv1.in_channels == 4`.
- C: `fc.out_features == 1`.
- B: forward on `(2, 4, 90, 90)` → `(2, 1)`.
- B: forward on `(1, 4, 120, 120)` → `(1, 1)`.
- B (`slow`): `pretrained=True` loads `IMAGENET1K_V1`; `conv1` is replaced so
  `conv1.weight.shape[1] == 4` (guards against accidental re-use of
  original ImageNet conv1 weights).

**`test_segmenter_unet.py`**
- B: `build_segmenter(in_channels=4, n_classes=1)` returns a `UNet`.
- C: forward on `(2, 4, 120, 120)` → `(2, 1, 120, 120)`.
- C: forward on `(2, 4, 90, 90)` → `(2, 1, 90, 90)`.
- B: `bilinear=False` (ConvTranspose path) produces correct shape.
- C (parameterized): size-preserving for crop ∈ {90, 120}.

### 4.5 `training/`

**`test_classification_module.py`**
- B: `__init__` saves hparams, builds net, creates metrics.
- C: forward on tiny batch returns shape `(B, 1)`.
- C: `_shared_step` returns scalar loss requiring grad.
- B: `training_step` / `validation_step` / `test_step` emit the expected
  log keys (`train/loss`, `train/acc`, `val/loss`, `val/acc`, `val/auc`,
  `test/acc`, `test/auc`) — verified by running a `Trainer(fast_dev_run=True)`
  against a trivial DataModule and inspecting `trainer.callback_metrics`.
- B: `configure_optimizers` — `scheduler="none"` returns an optimizer; `"plateau"`
  returns a dict with `monitor`; `"cosine"` returns a dict.
- C (`slow`): overfitting sanity — 20 SGD steps on a fixed 2-sample batch with
  `pretrained=False`; assert `final_loss < initial_loss * 0.95` and
  `loss_at_step_20 < loss_at_step_0`.

**`test_segmentation_module.py`**
- B: `_shared_step` reshapes `y` from `(B, H, W)` to `(B, 1, H, W)`.
- C: forward output shape matches mask shape.
- C: `val/img_acc` uses any-pixel image-level presence — verified with a
  batch containing one hit and one miss.
- B: three `configure_optimizers` scheduler branches covered.
- C (`slow`): overfitting sanity loop identical in spirit to the classifier test.

### 4.6 `evaluation/`

**`test_classification_metrics.py`**
- B: `plot_confusion_matrix(tp=3, tn=5, fp=1, fn=2, out_path=tmp)` writes
  a valid PNG (magic bytes `\x89PNG`).
- B: `plot_roc_curve([0.1, 0.9, 0.4, 0.8], [0, 1, 0, 1], tmp)` writes PNG.
- B: all-one-class labels cover the `ValueError` fallback branch producing
  `auc=nan`.

**`test_segmentation_metrics.py`**
- B: both plot helpers write valid PNGs.
- B: empty `ious=[]` / `ratios=[]` do not raise.

### 4.7 `cli/`

**`test_train_argparse.py`**
- B: `--help` exits 0; missing `--config` exits non-zero.
- B: multiple `--override` flags accumulate as a list.
- B: dispatch — classification config routes through `_build_classification`
  (monkeypatch `L.Trainer.fit` and inspect the module type passed in).

**`test_eval_argparse.py`**
- B: `--config` and `--ckpt` are required.
- B: `--out-dir` override respected; default equals
  `<output_dir>/<experiment_name>/eval`.

### 4.8 `scripts/prepare_dataset.py`

**`test_prepare_dataset_helpers.py`**
- B: `site_id_from_stem("ghana_2018-01-01")` → `"ghana"`.
- C: `json_url_to_tif_basename("/data/upload/1-site_2024:01:01Z.png")` strips
  the `/data/upload/1-` prefix, replaces `:` with `_`, swaps `.png` → `.tif`.
- B: `link_or_copy` in `copy` mode.
- B (`skipif` on Windows): `link_or_copy` in `hardlink` mode.
- B: invalid mode raises `ValueError`.

### 4.9 Integration

**`test_classification_datamodule.py`**
- B: `setup(stage="fit")` builds `train_ds` + `val_ds`; `"test"` builds `test_ds`.
- B: `train_dataloader()` batch has `img.shape == (B, 4, crop, crop)`, `lbl` is a bool tensor.
- B: `val_dataloader` uses `balance="none"`.
- C: `num_workers` is coerced to 0 when `platform.system()` returns `"Windows"`
  (monkeypatched).
- B: `save_hyperparameters` populates `self.hparams`.

**`test_segmentation_datamodule.py`**
- Analogous setup/dataloader tests; batch shapes are `(B, 4, crop, crop)` +
  `fpt (B, crop, crop)`.
- B: `test_dataloader` runs even when `test` is the only stage set up.

**`test_module_consumes_batch.py`** (both tasks)
- Pulls one real batch from each DataModule's train loader and runs it
  through the matching LightningModule's `training_step`. Asserts finite
  scalar loss with `requires_grad=True`.

**`test_config_to_trainer.py`**
- YAML → `load_config` → `_build_classification`/`_build_segmentation` →
  `trainer.fit` with `fast_dev_run=True`. Verifies a checkpoint file is
  written under `<output_dir>/<experiment>/version_0/checkpoints/`.

**`test_prepare_dataset_script.py`**
- `build_synthetic_zenodo_source` → shell out to
  `python scripts/prepare_dataset.py --source … --output … --mode copy` →
  assert resulting splits, verify train/val/test sites are disjoint, every
  label JSON has a matching `positive/*.tif`. Covers `--dry-run`, invalid
  ratio sums, missing source dir.

### 4.10 E2E

**`test_fast_dev_run.py`**
- Both tasks, YAML configs loaded with `trainer.fast_dev_run=True`,
  `batch_size=2`, `num_workers=0`, `paths.data_root=<synthetic>`; assert
  exit code 0. Supersedes `scripts/smoketest_fast_dev_run.py` (the
  standalone script either stays as a thin wrapper or is removed — decided
  during implementation).

**`test_train_eval_cycle.py`**
- `max_epochs=1`, `fast_dev_run=False`; then invoke eval CLI with the
  resulting `last.ckpt`; assert classification plots (`confusion_matrix.png`,
  `roc_curve.png`) and segmentation plots (`iou_distribution.png`,
  `area_ratio_distribution.png`) exist and are non-empty.

**`test_checkpoint_roundtrip.py`**
- Save a trained module checkpoint → `load_from_checkpoint` → forward pass
  on the same input is bit-identical.

### Tally

≈ 97 unit + 18 integration + 3 e2e ≈ **~120 tests**.

## 5. Cross-Cutting Concerns

- **Determinism.** Autouse `_deterministic` fixture seeds all RNGs per test;
  `cudnn.deterministic=True` for the session.
- **Windows.** `symlink`/`hardlink` prepare-dataset modes skipped; DataModule
  `num_workers` coercion tested via monkeypatch; all paths `pathlib.Path`;
  `rmtree(..., ignore_errors=True)` on teardown.
- **CUDA.** Session hook auto-skips `gpu` marker when CUDA unavailable. One
  GPU-forward test per module; everything else CPU.
- **Network.** Never hit at runtime. Pretrained weights pre-cached locally
  and restored via `actions/cache` in CI.
- **Time budget.** Default suite target < 60s; slow + default < ~2 min; e2e
  + default < ~3 min. Enforced per-test via `pytest-timeout` (default 30s;
  e2e override 300s).
- **Coverage gate.** `--cov-fail-under=80` on `src/smoke_detection/`, checked
  after the slow tier runs with `--cov-append`.
- **Lightning noise.** `PYTORCH_LIGHTNING_LOG_LEVEL=ERROR` for the session;
  TensorBoard output under `tmp_path`, never the repo's `lightning_logs/`.
- **Flaky guards.**
  - Overfitting sanity: 20 steps, assert `final < 0.95 * initial` AND
    `final < initial`. Fixed seed.
  - Polygon rasterization area: ±10% tolerance.
  - Plot tests inspect PNG magic bytes only, not pixel content.
- **Every `raise`** in `src/` is exercised by at least one test, tracked as
  a checklist during implementation.

## 6. CI Changes

### 6.1 `.github/workflows/ci.yml`

- **`test` job** (Ubuntu, Py 3.11 + 3.12 matrix)
  - Add `actions/cache@v4` step keyed on `resnet50-imagenet-v1` pointing at
    `~/.cache/torch/hub/checkpoints`.
  - Cache-miss warm step:
    ```yaml
    - name: Warm torchvision weights cache
      run: uv run python -c "from torchvision.models import resnet50, ResNet50_Weights; resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)"
    ```
  - Replace the `Pytest` step with:
    ```yaml
    - name: Pytest (fast tier)
      run: uv run pytest -m "not slow and not e2e and not gpu"
    - name: Pytest (slow tier + coverage gate)
      run: uv run pytest -m "slow" --cov-append --cov-fail-under=80
    ```

- **New `e2e` job** (Ubuntu, Py 3.12 only, `needs: test`)
  - Same uv + Python + weight-cache setup.
  - `uv run pytest -m "e2e" --timeout=300`.

- **`lint` and `build` jobs unchanged.**

### 6.2 `pyproject.toml`

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

### 6.3 Makefile

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

### 6.4 `CONTRIBUTING.md`

Add a short **Tests** subsection describing the marker scheme, the Make
targets, and the note that the CI cache key (`resnet50-imagenet-v1`) must
be bumped if the torchvision weights URL changes.

## 7. Success Criteria

1. `make test` passes on Ubuntu (3.11 & 3.12) and Windows (3.12) from a
   cold checkout.
2. `make test-e2e` passes locally and in the new CI `e2e` job.
3. `make test-gpu` passes on the dev machine (manual; not in CI).
4. Coverage ≥ 80% line + branch on `src/smoke_detection/`, enforced in CI.
5. No test hits the network at runtime.
6. CI `test` job stays under 5 minutes per matrix cell.
7. Every `raise` in `src/` is exercised by at least one test.
8. Every public function/class in `src/smoke_detection/` has a corresponding
   test file under `tests/unit/<subpackage>/`.
9. Every pydantic schema has at least one accept and one reject test.
10. Every LightningModule step method has a shape + log-key test.
11. Behavioral invariants covered: normalization zero-centering,
    augmentation mask/image synchronization, U-Net size-preservation,
    overfitting loss drop.
12. The suite is deterministic — two back-to-back runs of `make test`
    produce identical pass/fail.
13. New modules under `src/smoke_detection/` map 1:1 to
    `tests/unit/<subpackage>/test_<module>.py`.
14. Assertion messages cite the code under test.
15. `scripts/smoketest_fast_dev_run.py` is either reduced to a thin wrapper
    around the pytest e2e case or removed in favor of it.

## 8. Open Implementation Questions (to resolve during the plan phase)

- Whether to keep `scripts/smoketest_fast_dev_run.py` at all (item 15
  above). Decision made during plan-writing.
- Exact numeric thresholds for overfitting-sanity loops will be tuned once
  empirical runs are available; the plan reserves `final < 0.95 * initial`
  as the conservative default.
