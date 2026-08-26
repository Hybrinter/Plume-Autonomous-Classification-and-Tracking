# Spec 1 — Paper-Parity Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Phase 1 §1.1 of the roadmap: the paper-parity baseline gating layer. Produces a torch-free `evaluation/parity.py` library, a `evaluation/audit.py` library validating `paper.yaml` against a committed audit doc, a consolidated figures callback that owns both train- and test-time figure generation, an end-to-end orchestrator script, a parameterized notebook, and a runbook for the GPU-box workflow.

**Architecture:** Library-first (Approach 2). `evaluation/parity.py` and `evaluation/audit.py` are pure Python (stdlib + pydantic + pyyaml). `scripts/report_parity.py` and `scripts/run_paper_parity.py` are thin CLI wrappers. `figures_callback.py` becomes the single source of truth for matplotlib output across `cli/train.py` and `cli/eval.py`. The orchestrator runs train (subprocess) → test (in-process with callback) → parity-report in one command per task.

**Tech Stack:** Python 3.11+, PyTorch + Lightning + torchmetrics, pydantic v2, pyyaml, ruff, pytest, jupyter (notebooks extra), uv for env management.

**Spec:** [`docs/superpowers/specs/2026-04-25-spec-1-paper-parity-design.md`](../specs/2026-04-25-spec-1-paper-parity-design.md)

**Branch:** `phase-1-generalization` (current).

**Conventions:**
- All test/lint commands prefixed with `uv run`.
- All Python files start with `"""docstring"""\n\nfrom __future__ import annotations\n\n`.
- Pydantic v2 with `model_config = ConfigDict(extra="forbid")`.
- Tests live under `tests/{unit,integration,e2e}/<package_path>/test_*.py`.
- Markers `slow`, `gpu`, `e2e` exist; default test run excludes them. New unit tests use no marker. New integration tests covering CLI flows use `@pytest.mark.e2e` so they don't run by default.
- Pre-commit runs ruff + ruff-format + trailing-whitespace + end-of-file-fixer + check-yaml. Run `uv run pre-commit run --files <changed>` before each commit if the hooks are installed; otherwise `uv run ruff format <files> && uv run ruff check --fix <files>` is sufficient.

---

## Task 1: `evaluation/parity.py` — Pydantic types and threshold constants

**Files:**
- Create: `src/smoke_detection/evaluation/parity.py`
- Test: `tests/unit/evaluation/test_parity.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/evaluation/test_parity.py`:

```python
"""Unit tests for evaluation.parity."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from smoke_detection.evaluation.parity import (
    PAPER,
    TOLERANCE,
    DIRECTION,
    ParityMetric,
    ParityOverall,
    ParityProvenance,
    ParityReport,
)


def test_paper_tolerance_direction_keys_align():
    """PAPER, TOLERANCE, DIRECTION must agree on tasks and metric keys."""
    assert set(PAPER) == {"classification", "segmentation"}
    assert set(TOLERANCE) == set(PAPER)
    assert set(DIRECTION) == set(PAPER)
    for task in PAPER:
        assert set(PAPER[task]) == set(TOLERANCE[task]) == set(DIRECTION[task]), task


def test_parity_metric_round_trip():
    m = ParityMetric(
        value=0.927, paper=0.943, delta=-0.016,
        tolerance=0.02, direction="higher_better", **{"pass": True},
    )
    blob = m.model_dump_json(by_alias=True)
    restored = ParityMetric.model_validate_json(blob)
    assert restored == m
    assert '"pass":' in blob  # alias used in JSON


def test_parity_metric_rejects_extra():
    with pytest.raises(ValidationError):
        ParityMetric(
            value=0.5, paper=None, delta=None,
            tolerance=None, direction="higher_better",
            **{"pass": None}, junk_field=1,
        )


def test_parity_overall_round_trip():
    o = ParityOverall(passed_count=2, failed_count=0, ungated_count=1, **{"pass": True})
    blob = o.model_dump_json(by_alias=True)
    restored = ParityOverall.model_validate_json(blob)
    assert restored == o


def test_parity_report_round_trip():
    from datetime import datetime, timezone
    report = ParityReport(
        task="classification",
        produced_at=datetime(2026, 4, 25, 22, 14, 7, tzinfo=timezone.utc),
        provenance=ParityProvenance(
            git_commit="abc123",
            git_dirty=False,
            config_path="configs/classification/paper.yaml",
            config_sha256="deadbeef",
            checkpoint_path=None,
            checkpoint_sha256=None,
            dataset_root="/tmp/data",
            dataset_file_count={"train": 1, "val": 1, "test": 1},
        ),
        metrics={
            "test_accuracy": ParityMetric(
                value=0.927, paper=0.943, delta=-0.016,
                tolerance=0.02, direction="higher_better", **{"pass": True},
            ),
        },
        overall=ParityOverall(passed_count=1, failed_count=0, ungated_count=0, **{"pass": True}),
    )
    blob = report.model_dump_json(by_alias=True)
    restored = ParityReport.model_validate_json(blob)
    assert restored == report
    parsed = json.loads(blob)
    assert parsed["overall"]["pass"] is True
    assert parsed["metrics"]["test_accuracy"]["pass"] is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/evaluation/test_parity.py -v
```

Expected: collection error / ImportError on `smoke_detection.evaluation.parity` — the module doesn't exist yet.

- [ ] **Step 3: Create `src/smoke_detection/evaluation/parity.py`**

```python
"""Parity-report library — types, threshold constants, evaluators, JSON writer.

Pure Python: no torch/lightning at import time. Callers pass observed
metrics as a Mapping[str, float]; this module returns a serializable
ParityReport. This boundary lets §1.6's ablation runner reuse the
library without shape contortions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---- Reference baselines (Mommert et al. 2020, NeurIPS workshop) ----------

PAPER: dict[str, dict[str, float | None]] = {
    "classification": {
        "test_accuracy": 0.943,   # paper abstract
        "test_auc":      None,    # not reported
    },
    "segmentation": {
        "test_iou":                  0.608,
        "test_img_accuracy":         0.940,
        "mean_abs_area_ratio_error": 0.056,
    },
}

TOLERANCE: dict[str, dict[str, float | None]] = {
    "classification": {
        "test_accuracy": 0.02,
        "test_auc":      None,
    },
    "segmentation": {
        "test_iou":                  0.03,
        "test_img_accuracy":         0.02,
        "mean_abs_area_ratio_error": 0.03,
    },
}

DIRECTION: dict[str, dict[str, str]] = {
    "classification": {
        "test_accuracy": "higher_better",
        "test_auc":      "higher_better",
    },
    "segmentation": {
        "test_iou":                  "higher_better",
        "test_img_accuracy":         "higher_better",
        "mean_abs_area_ratio_error": "lower_better",
    },
}

Direction = Literal["higher_better", "lower_better"]


# ---- Schema --------------------------------------------------------------

class ParityMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    value:     float | None
    paper:     float | None
    delta:     float | None
    tolerance: float | None
    direction: Direction = "higher_better"
    pass_:     bool | None = Field(alias="pass")


class ParityProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    git_commit:        str
    git_dirty:         bool
    config_path:       str
    config_sha256:     str
    checkpoint_path:   str | None
    checkpoint_sha256: str | None
    dataset_root:      str
    dataset_file_count: dict[str, int]


class ParityOverall(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    pass_:         bool = Field(alias="pass")
    passed_count:  int
    failed_count:  int
    ungated_count: int


class ParityReport(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    schema_version: int = 1
    task:           Literal["classification", "segmentation"]
    produced_at:    datetime
    provenance:     ParityProvenance
    metrics:        dict[str, ParityMetric]
    overall:        ParityOverall
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/evaluation/test_parity.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/smoke_detection/evaluation/parity.py tests/unit/evaluation/test_parity.py
uv run ruff check --fix src/smoke_detection/evaluation/parity.py tests/unit/evaluation/test_parity.py
git add src/smoke_detection/evaluation/parity.py tests/unit/evaluation/test_parity.py
git commit -m "feat(evaluation): add parity types + threshold constants

Pure-python schema for paper-parity reports. PAPER/TOLERANCE/DIRECTION
constants encode the Standard tier from Spec 1 (cls acc ±2pp, seg IoU
±0.03, seg img-acc ±2pp, area-ratio one-sided ≤ paper + 0.03).

No torch/lightning at import time — keeps the library reusable from
§1.6's ablation runner without shape contortions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `parity.evaluate_thresholds()` — per-metric gate logic

**Files:**
- Modify: `src/smoke_detection/evaluation/parity.py`
- Modify: `tests/unit/evaluation/test_parity.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/unit/evaluation/test_parity.py`:

```python
import math

from smoke_detection.evaluation.parity import evaluate_thresholds


def test_evaluate_thresholds_classification_pass():
    metrics, overall = evaluate_thresholds(
        "classification",
        {"test_accuracy": 0.927, "test_auc": 0.981},
    )
    assert overall.pass_ is True
    assert overall.passed_count == 1
    assert overall.failed_count == 0
    assert overall.ungated_count == 1
    acc = metrics["test_accuracy"]
    assert acc.value == 0.927
    assert acc.paper == 0.943
    assert acc.delta == pytest.approx(-0.016, abs=1e-9)
    assert acc.pass_ is True
    auc = metrics["test_auc"]
    assert auc.value == 0.981
    assert auc.paper is None
    assert auc.pass_ is None     # ungated


def test_evaluate_thresholds_classification_fail_below_tolerance():
    """Acc 0.90 vs paper 0.943, tolerance ±0.02 → fail."""
    metrics, overall = evaluate_thresholds(
        "classification", {"test_accuracy": 0.90, "test_auc": 0.95},
    )
    assert overall.pass_ is False
    assert overall.failed_count == 1
    assert metrics["test_accuracy"].pass_ is False


def test_evaluate_thresholds_segmentation_lower_better():
    """area-ratio: lower better. paper=0.056, tol=0.03 → cap at 0.086."""
    metrics_pass, _ = evaluate_thresholds(
        "segmentation",
        {"test_iou": 0.61, "test_img_accuracy": 0.94, "mean_abs_area_ratio_error": 0.080},
    )
    assert metrics_pass["mean_abs_area_ratio_error"].pass_ is True

    metrics_fail, overall_fail = evaluate_thresholds(
        "segmentation",
        {"test_iou": 0.61, "test_img_accuracy": 0.94, "mean_abs_area_ratio_error": 0.090},
    )
    assert metrics_fail["mean_abs_area_ratio_error"].pass_ is False
    assert overall_fail.pass_ is False

    metrics_better, _ = evaluate_thresholds(
        "segmentation",
        {"test_iou": 0.61, "test_img_accuracy": 0.94, "mean_abs_area_ratio_error": 0.020},
    )
    assert metrics_better["mean_abs_area_ratio_error"].pass_ is True
    assert metrics_better["mean_abs_area_ratio_error"].delta == pytest.approx(-0.036, abs=1e-9)


def test_evaluate_thresholds_missing_metric_is_ungated_with_value_none():
    metrics, overall = evaluate_thresholds("classification", {"test_accuracy": 0.95})
    assert metrics["test_auc"].value is None
    assert metrics["test_auc"].pass_ is None
    assert overall.ungated_count == 1


def test_evaluate_thresholds_nan_observed_is_failure():
    metrics, overall = evaluate_thresholds(
        "classification",
        {"test_accuracy": float("nan"), "test_auc": 0.5},
    )
    assert math.isnan(metrics["test_accuracy"].value)  # type: ignore[arg-type]
    assert metrics["test_accuracy"].delta is None
    assert metrics["test_accuracy"].pass_ is False
    assert overall.failed_count == 1
    assert overall.pass_ is False


def test_evaluate_thresholds_unknown_observed_key_dropped():
    """A key not in PAPER[task] doesn't appear in the report."""
    metrics, overall = evaluate_thresholds(
        "classification",
        {"test_accuracy": 0.95, "test_auc": 0.99, "test/loss": 0.1},
    )
    assert "test/loss" not in metrics
    assert overall.passed_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/evaluation/test_parity.py -v
```

Expected: 6 new tests fail with `ImportError: cannot import name 'evaluate_thresholds'`.

- [ ] **Step 3: Add `evaluate_thresholds` to `parity.py`**

Append to `src/smoke_detection/evaluation/parity.py`:

```python
import logging
import math
from typing import Mapping

log = logging.getLogger(__name__)


def _gate(value: float, paper: float, tolerance: float, direction: Direction) -> bool:
    """Return True if `value` is within tolerance of `paper`, given direction."""
    if direction == "lower_better":
        # one-sided: only penalize when value > paper + tolerance
        return value <= paper + tolerance
    # higher_better: two-sided absolute tolerance
    return abs(value - paper) <= tolerance


def evaluate_thresholds(
    task: Literal["classification", "segmentation"],
    observed: Mapping[str, float],
) -> tuple[dict[str, ParityMetric], ParityOverall]:
    """Compare observed metrics against PAPER+TOLERANCE for `task`.

    Keys in `observed` not in PAPER[task] are dropped (logged at debug).
    Keys in PAPER[task] not in `observed` get value=None, pass=None, contribute
    to ungated_count.
    NaN/inf observed values: pass=False (treated as bug, not parity gap).
    """
    paper_table = PAPER[task]
    tol_table = TOLERANCE[task]
    dir_table = DIRECTION[task]

    metrics: dict[str, ParityMetric] = {}
    passed = failed = ungated = 0

    for key in paper_table:
        paper = paper_table[key]
        tolerance = tol_table[key]
        direction = dir_table[key]
        observed_val = observed.get(key)

        if observed_val is None:
            metrics[key] = ParityMetric(
                value=None, paper=paper, delta=None,
                tolerance=tolerance, direction=direction, **{"pass": None},
            )
            log.warning("metric %r expected for %s but not observed", key, task)
            ungated += 1
            continue

        if isinstance(observed_val, float) and not math.isfinite(observed_val):
            metrics[key] = ParityMetric(
                value=observed_val, paper=paper, delta=None,
                tolerance=tolerance, direction=direction, **{"pass": False},
            )
            failed += 1
            continue

        delta = (observed_val - paper) if paper is not None else None
        if paper is None or tolerance is None:
            pass_ = None
            ungated += 1
        else:
            pass_ = _gate(observed_val, paper, tolerance, direction)
            if pass_:
                passed += 1
            else:
                failed += 1

        metrics[key] = ParityMetric(
            value=observed_val, paper=paper, delta=delta,
            tolerance=tolerance, direction=direction, **{"pass": pass_},
        )

    for key in observed:
        if key not in paper_table:
            log.debug("observed metric %r not in PAPER[%s]; dropped", key, task)

    overall = ParityOverall(
        passed_count=passed, failed_count=failed, ungated_count=ungated,
        **{"pass": failed == 0},
    )
    return metrics, overall
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/evaluation/test_parity.py -v
```

Expected: 11 passed total.

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/smoke_detection/evaluation/parity.py tests/unit/evaluation/test_parity.py
uv run ruff check --fix src/smoke_detection/evaluation/parity.py tests/unit/evaluation/test_parity.py
git add src/smoke_detection/evaluation/parity.py tests/unit/evaluation/test_parity.py
git commit -m "feat(evaluation): add evaluate_thresholds() with one-sided gate

Per Spec 1 question 3: lower-is-better metrics (mean_abs_area_ratio_error)
get a one-sided gate (value ≤ paper + tolerance), higher-is-better
metrics get a two-sided absolute tolerance.

NaN/inf observed values fail hard rather than going ungated — almost
always a bug, not a parity gap; silent ungating would hide it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `parity.py` — provenance helpers (`hash_file`, `git_state`, `gather_provenance`)

**Files:**
- Modify: `src/smoke_detection/evaluation/parity.py`
- Modify: `tests/unit/evaluation/test_parity.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/unit/evaluation/test_parity.py`:

```python
import subprocess
from pathlib import Path

from smoke_detection.evaluation.parity import (
    gather_provenance,
    git_state,
    hash_file,
)


def test_hash_file_deterministic(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_bytes(b"hello world")
    h1 = hash_file(p)
    h2 = hash_file(p)
    assert h1 == h2
    assert len(h1) == 64
    # known SHA-256("hello world")
    assert h1 == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_hash_file_changes_on_content_change(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_bytes(b"hello")
    h1 = hash_file(p)
    p.write_bytes(b"world")
    h2 = hash_file(p)
    assert h1 != h2


def test_git_state_returns_tuple():
    """git_state must always return (str, bool) and never raise."""
    commit, dirty = git_state()
    assert isinstance(commit, str)
    assert isinstance(dirty, bool)


def test_git_state_handles_missing_git(tmp_path: Path, monkeypatch):
    """If git binary not on PATH, returns ('unknown', True), no exception."""
    monkeypatch.setenv("PATH", str(tmp_path))   # empty PATH
    commit, dirty = git_state()
    assert commit == "unknown"
    assert dirty is True


def test_gather_provenance_round_trip(tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("task: classification\n")
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"\x00" * 16)
    data_root = tmp_path / "data"
    data_root.mkdir()
    prov = gather_provenance(
        config_path=cfg,
        checkpoint_path=ckpt,
        dataset_root=data_root,
        dataset_file_count={"train": 8, "val": 2, "test": 2},
    )
    assert prov.config_path == str(cfg)
    assert prov.config_sha256 == hash_file(cfg)
    assert prov.checkpoint_sha256 == hash_file(ckpt)
    assert prov.dataset_file_count == {"train": 8, "val": 2, "test": 2}
    assert prov.dataset_root == str(data_root)
    assert isinstance(prov.git_commit, str)


def test_gather_provenance_no_checkpoint(tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("task: classification\n")
    prov = gather_provenance(
        config_path=cfg,
        checkpoint_path=None,
        dataset_root=tmp_path,
        dataset_file_count={"train": 0, "val": 0, "test": 0},
    )
    assert prov.checkpoint_path is None
    assert prov.checkpoint_sha256 is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/evaluation/test_parity.py -v
```

Expected: 6 new tests fail with ImportError on `gather_provenance`/`git_state`/`hash_file`.

- [ ] **Step 3: Append provenance helpers to `parity.py`**

```python
import hashlib
import shutil
import subprocess
from pathlib import Path


def hash_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hex digest of file contents. Streams in chunks for large files."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def git_state() -> tuple[str, bool]:
    """Return (commit_sha, is_dirty). Returns ('unknown', True) on any failure
    (no git binary, not in a repo, etc.) — never raises."""
    if shutil.which("git") is None:
        return "unknown", True
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout
        return commit, bool(status.strip())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return "unknown", True


def gather_provenance(
    config_path: Path,
    checkpoint_path: Path | None,
    dataset_root: Path,
    dataset_file_count: dict[str, int],
) -> ParityProvenance:
    """Build a ParityProvenance: hash files, capture git state, normalize paths."""
    commit, dirty = git_state()
    return ParityProvenance(
        git_commit=commit,
        git_dirty=dirty,
        config_path=str(config_path),
        config_sha256=hash_file(config_path),
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        checkpoint_sha256=hash_file(checkpoint_path) if checkpoint_path is not None else None,
        dataset_root=str(dataset_root),
        dataset_file_count=dataset_file_count,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/evaluation/test_parity.py -v
```

Expected: 17 passed total.

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/smoke_detection/evaluation/parity.py tests/unit/evaluation/test_parity.py
uv run ruff check --fix src/smoke_detection/evaluation/parity.py tests/unit/evaluation/test_parity.py
git add src/smoke_detection/evaluation/parity.py tests/unit/evaluation/test_parity.py
git commit -m "feat(evaluation): add provenance helpers (hash_file, git_state, gather_provenance)

git_state never raises — returns ('unknown', True) on any failure.
This lets the orchestrator produce a parity report even from
machines without git on PATH (some Orin SD images don't ship it).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `parity.py` — `LIGHTNING_KEY_MAP` + `translate_lightning_metrics`

**Files:**
- Modify: `src/smoke_detection/evaluation/parity.py`
- Modify: `tests/unit/evaluation/test_parity.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/unit/evaluation/test_parity.py`:

```python
from smoke_detection.evaluation.parity import (
    LIGHTNING_KEY_MAP,
    translate_lightning_metrics,
)


def test_lightning_key_map_covers_known_module_outputs():
    """All Lightning metric keys produced by ClassificationModule + SegmentationModule
    test_step must be in LIGHTNING_KEY_MAP."""
    expected_keys = {"test/acc", "test/auc", "test/iou", "test/img_acc"}
    assert expected_keys.issubset(set(LIGHTNING_KEY_MAP))


def test_translate_lightning_metrics_renames():
    raw = {"test/acc": 0.927, "test/auc": 0.981}
    result = translate_lightning_metrics(raw)
    assert result == {"test_accuracy": 0.927, "test_auc": 0.981}


def test_translate_passes_through_canonical_keys():
    """Already-canonical keys (like mean_abs_area_ratio_error) pass through."""
    raw = {"test/iou": 0.6, "mean_abs_area_ratio_error": 0.05}
    result = translate_lightning_metrics(raw)
    assert result == {"test_iou": 0.6, "mean_abs_area_ratio_error": 0.05}


def test_translate_drops_unknown_keys():
    """Unknown Lightning-style keys (e.g. train/loss leaked into test) are dropped."""
    raw = {"test/acc": 0.9, "train/loss": 0.5, "epoch": 99.0}
    result = translate_lightning_metrics(raw)
    assert result == {"test_accuracy": 0.9}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/evaluation/test_parity.py -v
```

Expected: 4 new tests fail with ImportError.

- [ ] **Step 3: Append translation logic to `parity.py`**

```python
LIGHTNING_KEY_MAP: dict[str, str] = {
    "test/acc":     "test_accuracy",
    "test/auc":     "test_auc",
    "test/iou":     "test_iou",
    "test/img_acc": "test_img_accuracy",
    # mean_abs_area_ratio_error is computed by callers (segmentation only;
    # not exposed by Trainer.test) and passed in directly under that name.
}

# Set of canonical keys that should pass through translate_lightning_metrics
# unchanged. (Anything already in PAPER[any_task] qualifies.)
_CANONICAL_KEYS: set[str] = {
    key for table in PAPER.values() for key in table
}


def translate_lightning_metrics(raw: Mapping[str, float]) -> dict[str, float]:
    """Apply LIGHTNING_KEY_MAP; pass through already-canonical keys; drop unknowns.

    Unknown keys (not in the map and not canonical) are dropped with a debug log
    rather than raising — Lightning's callback_metrics often contain extras
    like 'epoch' or 'step' that aren't relevant to parity.
    """
    out: dict[str, float] = {}
    for key, value in raw.items():
        if key in LIGHTNING_KEY_MAP:
            out[LIGHTNING_KEY_MAP[key]] = float(value)
        elif key in _CANONICAL_KEYS:
            out[key] = float(value)
        else:
            log.debug("translate_lightning_metrics: dropping unknown key %r", key)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/evaluation/test_parity.py -v
```

Expected: 21 passed total.

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/smoke_detection/evaluation/parity.py tests/unit/evaluation/test_parity.py
uv run ruff check --fix src/smoke_detection/evaluation/parity.py tests/unit/evaluation/test_parity.py
git add src/smoke_detection/evaluation/parity.py tests/unit/evaluation/test_parity.py
git commit -m "feat(evaluation): translate Lightning metric keys to canonical names

Lightning logs metrics with slash-prefixed keys (test/acc) which are
awkward in JSON and Python identifiers. Spec 1 schema uses canonical
snake_case names. translate_lightning_metrics bridges the two.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `parity.py` — `write_parity_report` + `format_table`

**Files:**
- Modify: `src/smoke_detection/evaluation/parity.py`
- Modify: `tests/unit/evaluation/test_parity.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/unit/evaluation/test_parity.py`:

```python
from smoke_detection.evaluation.parity import (
    format_table,
    write_parity_report,
)


def test_write_parity_report_classification_pass(tmp_path: Path):
    cfg = tmp_path / "cls.yaml"
    cfg.write_text("task: classification\n")
    ckpt = tmp_path / "m.ckpt"
    ckpt.write_bytes(b"weights")
    out = tmp_path / "parity.json"

    report = write_parity_report(
        task="classification",
        observed={"test_accuracy": 0.93, "test_auc": 0.98},
        config_path=cfg,
        checkpoint_path=ckpt,
        dataset_root=tmp_path,
        dataset_file_count={"train": 1, "val": 1, "test": 1},
        out_path=out,
    )
    assert report.task == "classification"
    assert report.overall.pass_ is True

    # JSON file written with both 'pass' alias and canonical keys
    raw = json.loads(out.read_text())
    assert raw["task"] == "classification"
    assert raw["schema_version"] == 1
    assert raw["overall"]["pass"] is True
    assert raw["metrics"]["test_accuracy"]["pass"] is True
    assert raw["provenance"]["config_sha256"] == hash_file(cfg)


def test_write_parity_report_round_trip(tmp_path: Path):
    cfg = tmp_path / "seg.yaml"
    cfg.write_text("task: segmentation\n")
    out = tmp_path / "parity.json"
    report = write_parity_report(
        task="segmentation",
        observed={"test_iou": 0.61, "test_img_accuracy": 0.94, "mean_abs_area_ratio_error": 0.05},
        config_path=cfg,
        checkpoint_path=None,
        dataset_root=tmp_path,
        dataset_file_count={"train": 1, "val": 1, "test": 1},
        out_path=out,
    )
    restored = ParityReport.model_validate_json(out.read_text())
    assert restored == report


def test_format_table_includes_metric_names_and_deltas(tmp_path: Path):
    report = write_parity_report(
        task="classification",
        observed={"test_accuracy": 0.95},
        config_path=Path(__file__),
        checkpoint_path=None,
        dataset_root=Path(__file__).parent,
        dataset_file_count={"train": 0, "val": 0, "test": 0},
        out_path=tmp_path / "_unused_format_table_test.json",
    )
    text = format_table(report)
    assert "test_accuracy" in text
    assert "0.9500" in text or "0.950" in text
    assert "+0.0070" in text or "0.007" in text   # 0.95 - 0.943
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/evaluation/test_parity.py -v
```

Expected: 3 new tests fail with ImportError.

- [ ] **Step 3: Append the writer + table formatter**

```python
import json
from datetime import datetime, timezone


def write_parity_report(
    task: Literal["classification", "segmentation"],
    observed: Mapping[str, float],
    config_path: Path,
    checkpoint_path: Path | None,
    dataset_root: Path,
    dataset_file_count: dict[str, int],
    out_path: Path,
) -> ParityReport:
    """Build, serialize, and write a ParityReport to `out_path`. Returns the report."""
    metrics, overall = evaluate_thresholds(task, observed)
    provenance = gather_provenance(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        dataset_root=dataset_root,
        dataset_file_count=dataset_file_count,
    )
    report = ParityReport(
        task=task,
        produced_at=datetime.now(tz=timezone.utc),
        provenance=provenance,
        metrics=metrics,
        overall=overall,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.model_dump_json(by_alias=True, indent=2))
    return report


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.4f}"


def format_table(report: ParityReport) -> str:
    """Human-readable stdout summary. Same shape as legacy report_parity.py."""
    rows: list[tuple[str, float | None, float | None]] = []
    for name, m in report.metrics.items():
        rows.append((name, m.value, m.paper))
    if not rows:
        return f"=== {report.task} parity ===\n(no metrics recorded)\n"
    w = max(len(r[0]) for r in rows)
    header = f"{'metric'.ljust(w)}   {'ours':>8}   {'paper':>8}   {'delta':>8}   {'pass':>5}"
    sep = "-" * (w + 38)
    lines = [
        f"=== {report.task} parity (vs. Mommert et al. 2020) ===",
        header,
        sep,
    ]
    for name, ours, paper in rows:
        m = report.metrics[name]
        delta_str = "—" if m.delta is None else f"{m.delta:+.4f}"
        if m.pass_ is None:
            pass_str = "—"
        else:
            pass_str = "PASS" if m.pass_ else "FAIL"
        lines.append(f"{name.ljust(w)}   {_fmt(ours):>8}   {_fmt(paper):>8}   {delta_str:>8}   {pass_str:>5}")
    lines.append(sep)
    overall_str = "PASS" if report.overall.pass_ else "FAIL"
    lines.append(
        f"OVERALL: {overall_str}  "
        f"(passed={report.overall.passed_count} failed={report.overall.failed_count} "
        f"ungated={report.overall.ungated_count})"
    )
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/evaluation/test_parity.py -v
```

Expected: 24 passed total.

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/smoke_detection/evaluation/parity.py tests/unit/evaluation/test_parity.py
uv run ruff check --fix src/smoke_detection/evaluation/parity.py tests/unit/evaluation/test_parity.py
git add src/smoke_detection/evaluation/parity.py tests/unit/evaluation/test_parity.py
git commit -m "feat(evaluation): add write_parity_report + format_table

The library is now feature-complete. write_parity_report glues the
threshold + provenance helpers and emits the JSON. format_table
preserves the stdout shape from legacy report_parity.py for human
review at the GPU-box terminal.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `configs/{cls,seg}/paper.yaml` — per-knob source comments

**Files:**
- Modify: `configs/classification/paper.yaml`
- Modify: `configs/segmentation/paper.yaml`

- [ ] **Step 1: Replace `configs/classification/paper.yaml` content**

```yaml
# Paper-replication config for classification (Mommert et al. 2020, 4-channel).
# Each knob carries a source attribution per docs/PAPER_PARITY_AUDIT.md.
# Source values: paper / hsg-aiml@64c806b / spec.
# Rows marked `verify-against-paper` are pending PDF cross-check.

task: classification
seed: 42                                        # source: spec
trainer:
  max_epochs: 100                               # source: hsg-aiml@64c806b (verify-against-paper)
  accelerator: auto
  devices: auto
  precision: "32"                               # source: hsg-aiml@64c806b (paper silent on dtype)
  deterministic: true
  log_every_n_steps: 10
paths:
  data_root: data
  output_dir: lightning_logs
  experiment_name: classification_4ch_resnet50_paper
optim:
  lr: 0.3                                       # source: hsg-aiml@64c806b (verify-against-paper)
  momentum: 0.7                                 # source: hsg-aiml@64c806b (unusual; verify-against-paper)
  weight_decay: 0.0                             # source: hsg-aiml@64c806b (verify-against-paper)
  scheduler: plateau                            # source: hsg-aiml@64c806b (verify-against-paper)
model:
  backbone: resnet50                            # source: hsg-aiml@64c806b (verify-against-paper)
  pretrained: true                              # source: hsg-aiml@64c806b (verify-against-paper)
  in_channels: 4                                # source: hsg-aiml@64c806b (verify-against-paper)
data:
  batch_size: 30                                # source: hsg-aiml@64c806b (verify-against-paper)
  num_workers: 4
  crop_size: 90                                 # source: hsg-aiml@64c806b (verify-against-paper)
  balance: upsample                             # source: hsg-aiml@64c806b (verify-against-paper)
```

- [ ] **Step 2: Replace `configs/segmentation/paper.yaml` content**

```yaml
# Paper-replication config for segmentation (Mommert et al. 2020, 4-channel).
# Each knob carries a source attribution per docs/PAPER_PARITY_AUDIT.md.
# Source values: paper / hsg-aiml@64c806b / spec.
# Rows marked `verify-against-paper` are pending PDF cross-check.

task: segmentation
seed: 42                                        # source: spec
trainer:
  max_epochs: 300                               # source: hsg-aiml@64c806b (verify-against-paper)
  accelerator: auto
  devices: auto
  precision: "32"                               # source: hsg-aiml@64c806b (paper silent on dtype)
  deterministic: true
  log_every_n_steps: 10
paths:
  data_root: data
  output_dir: lightning_logs
  experiment_name: segmentation_4ch_unet_paper
optim:
  lr: 0.7                                       # source: hsg-aiml@64c806b (verify-against-paper)
  momentum: 0.7                                 # source: hsg-aiml@64c806b (unusual; verify-against-paper)
  weight_decay: 0.0                             # source: hsg-aiml@64c806b (verify-against-paper)
  scheduler: plateau                            # source: hsg-aiml@64c806b (verify-against-paper)
model:
  architecture: unet                            # source: hsg-aiml@64c806b (verify-against-paper)
  in_channels: 4                                # source: hsg-aiml@64c806b (verify-against-paper)
  n_classes: 1                                  # source: hsg-aiml@64c806b
  bilinear: true                                # source: hsg-aiml@64c806b
data:
  batch_size: 60                                # source: hsg-aiml@64c806b (verify-against-paper)
  num_workers: 4
  crop_size: 90                                 # source: hsg-aiml@64c806b (verify-against-paper)
```

- [ ] **Step 3: Verify YAMLs still parse**

```bash
uv run python -c "from smoke_detection.configs.loader import load_config; cfg = load_config('configs/classification/paper.yaml'); print(cfg.optim.lr); cfg = load_config('configs/segmentation/paper.yaml'); print(cfg.optim.lr)"
```

Expected output:
```
0.3
0.7
```

- [ ] **Step 4: Run existing config tests**

```bash
uv run pytest tests/unit/configs tests/integration/test_config_to_trainer.py -v
```

Expected: all existing config tests still pass.

- [ ] **Step 5: Commit**

```bash
git add configs/classification/paper.yaml configs/segmentation/paper.yaml
git commit -m "chore(configs): annotate paper.yaml with per-knob source attribution

Each hyperparameter now declares its source (hsg-aiml@64c806b vs. paper
vs. spec). Rows marked 'verify-against-paper' are pending PDF
cross-check — runbook step in docs/RUNBOOK_paper_parity.md addresses
the cleanup pass when PDF access is restored.

Pure documentation; loader/parser is unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: `docs/PAPER_PARITY_AUDIT.md` — initial conservative content

**Files:**
- Create: `docs/PAPER_PARITY_AUDIT.md`

- [ ] **Step 1: Create the audit doc**

Write to `docs/PAPER_PARITY_AUDIT.md`:

```markdown
# Paper Parity Audit

Authoritative comparison of `configs/{classification,segmentation}/paper.yaml`
against (a) Mommert et al. 2020, NeurIPS Tackling Climate Change with ML
workshop and (b) the pre-Lightning HSG-AIML repo at commit `64c806b`.

**Source priority:** publication PDF first; HSG-AIML repo second; "?" if neither.

**Decision rule on disagreement:** prefer the publication; document the choice
in the `notes` column.

**Initial conservative state (2026-04-25):** the spec author did not have PDF
access at audit time. Every "could-be-in-paper" row is sourced from
`hsg-aiml@64c806b` and noted `verify-against-paper`. The runbook (`docs/RUNBOOK_paper_parity.md`)
includes a verification pass when PDF access is restored.

**Rendering convention** (used by `evaluation.audit._render`):
- Booleans render as `true` / `false`
- `None` renders as `null`
- Strings render unquoted
- Numbers via `repr` (so `0.3` renders as `0.3`, not `0.30000`)

## Classification (`configs/classification/paper.yaml`)

| section.key                            | yaml_value | paper_value | hsg_aiml_value | source       | notes |
|----------------------------------------|------------|-------------|----------------|--------------|-------|
| trainer.max_epochs                     | 100        | ?           | 100            | hsg-aiml     | verify-against-paper |
| trainer.precision                      | 32         | ?           | float32        | hsg-aiml     | paper silent on dtype |
| trainer.deterministic                  | true       | ?           | true           | hsg-aiml     | reproducibility flag |
| trainer.log_every_n_steps              | 10         | ?           | 10             | hsg-aiml     | logging cadence |
| optim.lr                               | 0.3        | ?           | 0.3            | hsg-aiml     | verify-against-paper |
| optim.momentum                         | 0.7        | ?           | 0.7            | hsg-aiml     | unusual; verify-against-paper |
| optim.weight_decay                     | 0.0        | ?           | 0.0            | hsg-aiml     | verify-against-paper |
| optim.scheduler                        | plateau    | ?           | plateau        | hsg-aiml     | verify-against-paper |
| model.backbone                         | resnet50   | ?           | resnet50       | hsg-aiml     | verify-against-paper |
| model.pretrained                       | true       | ?           | true           | hsg-aiml     | verify-against-paper |
| model.in_channels                      | 4          | ?           | 4              | hsg-aiml     | paper also reports 12-ch; verify-against-paper |
| data.batch_size                        | 30         | ?           | 30             | hsg-aiml     | verify-against-paper |
| data.num_workers                       | 4          | ?           | 4              | hsg-aiml     | runtime-tuned, not load-bearing |
| data.crop_size                         | 90         | ?           | 90             | hsg-aiml     | verify-against-paper |
| data.balance                           | upsample   | ?           | upsample       | hsg-aiml     | verify-against-paper |
| **gate**: test_accuracy                | tol ±0.02  | 0.943       | —              | spec         | per Spec 1 question 3 (Standard) |
| **gate**: test_auc                     | ungated    | —           | —              | spec         | not reported in publication |

## Segmentation (`configs/segmentation/paper.yaml`)

| section.key                            | yaml_value | paper_value | hsg_aiml_value | source       | notes |
|----------------------------------------|------------|-------------|----------------|--------------|-------|
| trainer.max_epochs                     | 300        | ?           | 300            | hsg-aiml     | verify-against-paper |
| trainer.precision                      | 32         | ?           | float32        | hsg-aiml     | paper silent on dtype |
| trainer.deterministic                  | true       | ?           | true           | hsg-aiml     | reproducibility flag |
| trainer.log_every_n_steps              | 10         | ?           | 10             | hsg-aiml     | logging cadence |
| optim.lr                               | 0.7        | ?           | 0.7            | hsg-aiml     | verify-against-paper |
| optim.momentum                         | 0.7        | ?           | 0.7            | hsg-aiml     | unusual; verify-against-paper |
| optim.weight_decay                     | 0.0        | ?           | 0.0            | hsg-aiml     | verify-against-paper |
| optim.scheduler                        | plateau    | ?           | plateau        | hsg-aiml     | verify-against-paper |
| model.architecture                     | unet       | ?           | unet           | hsg-aiml     | verify-against-paper |
| model.in_channels                      | 4          | ?           | 4              | hsg-aiml     | paper also reports 12-ch; verify-against-paper |
| model.n_classes                        | 1          | ?           | 1              | hsg-aiml     | binary mask |
| model.bilinear                         | true       | ?           | true           | hsg-aiml     | upsampling style |
| data.batch_size                        | 60         | ?           | 60             | hsg-aiml     | verify-against-paper |
| data.num_workers                       | 4          | ?           | 4              | hsg-aiml     | runtime-tuned, not load-bearing |
| data.crop_size                         | 90         | ?           | 90             | hsg-aiml     | verify-against-paper |
| **gate**: test_iou                     | tol ±0.03  | 0.608       | —              | spec         | per Spec 1 question 3 (Standard) |
| **gate**: test_img_accuracy            | tol ±0.02  | 0.940       | —              | spec         | per Spec 1 question 3 (Standard) |
| **gate**: mean_abs_area_ratio_error    | one-sided ≤ paper + 0.03 | 0.056 | — | spec       | per Spec 1 question 3 (lower-better) |
```

- [ ] **Step 2: Verify the doc parses cleanly as markdown**

```bash
uv run python -c "
text = open('docs/PAPER_PARITY_AUDIT.md').read()
assert '## Classification' in text
assert '## Segmentation' in text
assert 'verify-against-paper' in text
print('OK')
"
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add docs/PAPER_PARITY_AUDIT.md
git commit -m "docs: add PAPER_PARITY_AUDIT.md (conservative initial state)

Per Spec 1 question 5: per-knob audit table with publication-first /
hsg-aiml-fallback source priority. Initial state is conservative —
all 'could be in paper' rows marked 'verify-against-paper' until PDF
access is restored. Runbook documents the verification pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: `evaluation/audit.py` — types + `load_audit_table` parser

**Files:**
- Create: `src/smoke_detection/evaluation/audit.py`
- Test: `tests/unit/evaluation/test_audit.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/evaluation/test_audit.py`:

```python
"""Unit tests for evaluation.audit."""

from __future__ import annotations

from pathlib import Path

import pytest

from smoke_detection.evaluation.audit import (
    AuditRow,
    AuditTable,
    _render,
    load_audit_table,
)


SYNTH_MD = """\
# Test Audit

Some preamble text.

## Classification (`configs/classification/paper.yaml`)

| section.key | yaml_value | paper_value | hsg_aiml_value | source | notes |
|-------------|------------|-------------|----------------|--------|-------|
| optim.lr    | 0.3        | ?           | 0.3            | hsg-aiml | verify-against-paper |
| **gate**: test_accuracy | tol ±0.02 | 0.943 | — | spec | gate row |

## Segmentation (`configs/segmentation/paper.yaml`)

| section.key | yaml_value | paper_value | hsg_aiml_value | source | notes |
|-------------|------------|-------------|----------------|--------|-------|
| optim.lr    | 0.7        | ?           | 0.7            | hsg-aiml | verify-against-paper |
| **gate**: test_iou | tol ±0.03 | 0.608 | — | spec | seg gate |
"""


def test_render_basics():
    assert _render(True) == "true"
    assert _render(False) == "false"
    assert _render(None) == "null"
    assert _render("plateau") == "plateau"
    assert _render(0.3) == "0.3"
    assert _render(100) == "100"


def test_load_audit_table_parses_both_sections(tmp_path: Path):
    md = tmp_path / "audit.md"
    md.write_text(SYNTH_MD)
    tables = load_audit_table(md)
    assert set(tables) == {"classification", "segmentation"}
    assert isinstance(tables["classification"], AuditTable)
    assert tables["classification"].task == "classification"
    assert tables["segmentation"].task == "segmentation"


def test_load_audit_table_strips_gate_prefix(tmp_path: Path):
    md = tmp_path / "audit.md"
    md.write_text(SYNTH_MD)
    tables = load_audit_table(md)
    rows_by_key = {r.key: r for r in tables["classification"].rows}
    assert "test_accuracy" in rows_by_key
    assert rows_by_key["test_accuracy"].is_gate is True
    assert rows_by_key["optim.lr"].is_gate is False


def test_load_audit_table_preserves_row_fields(tmp_path: Path):
    md = tmp_path / "audit.md"
    md.write_text(SYNTH_MD)
    tables = load_audit_table(md)
    lr = next(r for r in tables["classification"].rows if r.key == "optim.lr")
    assert lr.yaml_value == "0.3"
    assert lr.paper_value == "?"
    assert lr.hsg_aiml_value == "0.3"
    assert lr.source == "hsg-aiml"
    assert lr.notes == "verify-against-paper"


def test_load_audit_table_rejects_bad_source_value(tmp_path: Path):
    bad = tmp_path / "audit.md"
    bad.write_text(
        "## Classification (`x`)\n\n"
        "| section.key | yaml_value | paper_value | hsg_aiml_value | source | notes |\n"
        "|---|---|---|---|---|---|\n"
        "| optim.lr | 0.3 | ? | 0.3 | random | x |\n"
    )
    with pytest.raises(Exception):  # pydantic ValidationError on Literal
        load_audit_table(bad)


def test_load_audit_table_against_committed_doc():
    """The committed audit doc must parse cleanly with both expected sections."""
    tables = load_audit_table(Path("docs/PAPER_PARITY_AUDIT.md"))
    assert set(tables) == {"classification", "segmentation"}
    cls_keys = {r.key for r in tables["classification"].rows}
    assert "optim.lr" in cls_keys
    assert "test_accuracy" in cls_keys
    seg_keys = {r.key for r in tables["segmentation"].rows}
    assert "test_iou" in seg_keys
    assert "mean_abs_area_ratio_error" in seg_keys
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/evaluation/test_audit.py -v
```

Expected: collection error / ImportError on `smoke_detection.evaluation.audit`.

- [ ] **Step 3: Create `src/smoke_detection/evaluation/audit.py`**

```python
"""Paper-parity audit: parse PAPER_PARITY_AUDIT.md and validate against paper.yaml.

Stdlib + pydantic + pyyaml only. Used by:
  - tests/unit/evaluation/test_audit.py (Spec 1)
  - scripts/run_paper_parity.py --strict-paper (Spec 1)
  - Phase 2 §2.2 CI gate (future)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict


_GATE_PREFIX = "**gate**:"


class AuditRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key:            str
    yaml_value:     str
    paper_value:    str
    hsg_aiml_value: str
    source:         Literal["paper", "hsg-aiml", "spec", "?"]
    notes:          str
    is_gate:        bool = False


class AuditTable(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: Literal["classification", "segmentation"]
    rows: list[AuditRow]


def _render(value: Any) -> str:
    """Canonical string rendering for YAML values, matching the audit-table column."""
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        return repr(value)
    return str(value)


_SECTION_RE = re.compile(r"^##\s+(Classification|Segmentation)\s*\(", re.MULTILINE)


def _split_sections(text: str) -> dict[str, str]:
    """Return {'classification': <body>, 'segmentation': <body>} given the full md."""
    matches = list(_SECTION_RE.finditer(text))
    out: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[name] = text[start:end]
    return out


def _parse_table(body: str) -> list[AuditRow]:
    """Find the first pipe-table in `body`; parse its rows into AuditRow objects."""
    lines = [ln.strip() for ln in body.splitlines()]
    table_lines: list[str] = []
    in_table = False
    for ln in lines:
        if ln.startswith("|"):
            table_lines.append(ln)
            in_table = True
        elif in_table:
            break  # table ended

    if len(table_lines) < 3:
        raise ValueError("audit table requires header + separator + at least one row")

    def split_row(row: str) -> list[str]:
        # strip leading/trailing pipe, split, strip cells
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        return cells

    header = split_row(table_lines[0])
    expected = {"section.key", "yaml_value", "paper_value", "hsg_aiml_value", "source", "notes"}
    if not expected.issubset(set(header)):
        raise ValueError(f"audit table header missing columns: {expected - set(header)}")
    col_idx = {name: i for i, name in enumerate(header)}

    rows: list[AuditRow] = []
    for raw in table_lines[2:]:  # skip separator
        cells = split_row(raw)
        if len(cells) != len(header):
            raise ValueError(f"audit table row has {len(cells)} cells, expected {len(header)}: {raw!r}")
        key = cells[col_idx["section.key"]]
        is_gate = key.startswith(_GATE_PREFIX)
        if is_gate:
            key = key[len(_GATE_PREFIX):].strip()
        rows.append(
            AuditRow(
                key=key,
                yaml_value=cells[col_idx["yaml_value"]],
                paper_value=cells[col_idx["paper_value"]],
                hsg_aiml_value=cells[col_idx["hsg_aiml_value"]],
                source=cells[col_idx["source"]],
                notes=cells[col_idx["notes"]],
                is_gate=is_gate,
            )
        )
    return rows


def load_audit_table(
    md_path: Path = Path("docs/PAPER_PARITY_AUDIT.md"),
) -> dict[str, AuditTable]:
    """Parse the markdown file into one AuditTable per task."""
    text = md_path.read_text(encoding="utf-8")
    sections = _split_sections(text)
    if not sections:
        raise ValueError(f"no '## Classification' / '## Segmentation' headers in {md_path}")
    out: dict[str, AuditTable] = {}
    for task_name, body in sections.items():
        rows = _parse_table(body)
        out[task_name] = AuditTable(task=task_name, rows=rows)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/evaluation/test_audit.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/smoke_detection/evaluation/audit.py tests/unit/evaluation/test_audit.py
uv run ruff check --fix src/smoke_detection/evaluation/audit.py tests/unit/evaluation/test_audit.py
git add src/smoke_detection/evaluation/audit.py tests/unit/evaluation/test_audit.py
git commit -m "feat(evaluation): add audit.py — load_audit_table + parser

Parses PAPER_PARITY_AUDIT.md into typed AuditTable objects. Strips the
'**gate**:' prefix from gate rows and sets is_gate=True. Stdlib markdown
parser (no third-party dep). Test verifies the committed audit doc
itself parses cleanly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: `audit.validate_against_yaml` + `validate_all`

**Files:**
- Modify: `src/smoke_detection/evaluation/audit.py`
- Modify: `tests/unit/evaluation/test_audit.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/unit/evaluation/test_audit.py`:

```python
import yaml as _yaml

from smoke_detection.evaluation.audit import (
    validate_against_yaml,
    validate_all,
)


def _write_minimal_paper_yaml(path: Path, lr: float = 0.3, momentum: float = 0.7) -> None:
    path.write_text(_yaml.safe_dump({
        "task": "classification",
        "seed": 42,
        "trainer": {"max_epochs": 100, "precision": "32", "deterministic": True, "log_every_n_steps": 10},
        "paths": {"data_root": "data", "output_dir": "lightning_logs",
                  "experiment_name": "exp"},
        "optim": {"lr": lr, "momentum": momentum, "weight_decay": 0.0, "scheduler": "plateau"},
        "model": {"backbone": "resnet50", "pretrained": True, "in_channels": 4},
        "data": {"batch_size": 30, "num_workers": 4, "crop_size": 90, "balance": "upsample"},
    }))


def test_validate_against_yaml_clean(tmp_path: Path):
    md = tmp_path / "audit.md"
    md.write_text(SYNTH_MD)
    yml = tmp_path / "paper.yaml"
    _write_minimal_paper_yaml(yml, lr=0.3)
    table = load_audit_table(md)["classification"]
    discrepancies = validate_against_yaml(table, yml)
    assert discrepancies == []


def test_validate_against_yaml_detects_value_drift(tmp_path: Path):
    md = tmp_path / "audit.md"
    md.write_text(SYNTH_MD)
    yml = tmp_path / "paper.yaml"
    _write_minimal_paper_yaml(yml, lr=0.5)   # disagrees with audit doc's 0.3
    table = load_audit_table(md)["classification"]
    discrepancies = validate_against_yaml(table, yml)
    assert len(discrepancies) == 1
    assert "optim.lr" in discrepancies[0]
    assert "0.3" in discrepancies[0] and "0.5" in discrepancies[0]


def test_validate_against_yaml_detects_missing_key(tmp_path: Path):
    md = tmp_path / "audit.md"
    md.write_text(SYNTH_MD)
    yml = tmp_path / "paper.yaml"
    yml.write_text(_yaml.safe_dump({"task": "classification"}))   # no optim.lr key
    table = load_audit_table(md)["classification"]
    discrepancies = validate_against_yaml(table, yml)
    assert any("optim.lr" in d for d in discrepancies)
    assert any("missing" in d.lower() for d in discrepancies)


def test_validate_against_yaml_skips_gate_rows(tmp_path: Path):
    """Gate rows aren't YAML knobs; validator skips them."""
    md = tmp_path / "audit.md"
    md.write_text(SYNTH_MD)
    yml = tmp_path / "paper.yaml"
    _write_minimal_paper_yaml(yml, lr=0.3)
    table = load_audit_table(md)["classification"]
    discrepancies = validate_against_yaml(table, yml)
    # Should not complain about test_accuracy not being a YAML key
    assert all("test_accuracy" not in d for d in discrepancies)


def test_validate_all_against_committed_yamls():
    """End-to-end: the committed audit doc + paper.yaml files agree."""
    discrepancies = validate_all()
    assert discrepancies == [], f"audit drift detected:\n" + "\n".join(discrepancies)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/evaluation/test_audit.py -v
```

Expected: 5 new tests fail with ImportError.

- [ ] **Step 3: Append validators to `audit.py`**

```python
def _resolve_dotted(data: dict[str, Any], dotted: str) -> Any:
    """Walk a nested dict by dotted key. Returns _MISSING sentinel if absent."""
    node: Any = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


_MISSING = object()


def validate_against_yaml(table: AuditTable, yaml_path: Path) -> list[str]:
    """For each non-gate row, assert table.row.yaml_value matches the rendered
    YAML value at table.row.key. Returns a list of discrepancy strings; empty = clean."""
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    discrepancies: list[str] = []
    for row in table.rows:
        if row.is_gate:
            continue
        actual = _resolve_dotted(data, row.key)
        if actual is _MISSING:
            discrepancies.append(
                f"{table.task} {row.key}: missing from {yaml_path} "
                f"(audit doc says {row.yaml_value!r})"
            )
            continue
        rendered = _render(actual)
        if rendered != row.yaml_value:
            discrepancies.append(
                f"{table.task} {row.key}: yaml has {rendered!r}, "
                f"audit doc says {row.yaml_value!r}"
            )
    return discrepancies


def validate_all(
    md_path: Path = Path("docs/PAPER_PARITY_AUDIT.md"),
    cls_yaml: Path = Path("configs/classification/paper.yaml"),
    seg_yaml: Path = Path("configs/segmentation/paper.yaml"),
) -> list[str]:
    """Convenience: load both tables, validate each against its YAML."""
    tables = load_audit_table(md_path)
    discrepancies: list[str] = []
    discrepancies.extend(validate_against_yaml(tables["classification"], cls_yaml))
    discrepancies.extend(validate_against_yaml(tables["segmentation"], seg_yaml))
    return discrepancies
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/evaluation/test_audit.py -v
```

Expected: 11 passed total.

If `test_validate_all_against_committed_yamls` fails, the audit doc and `paper.yaml` files have drifted — fix the YAML or the audit doc to align (per the rendering rules in `_render`), then re-run.

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/smoke_detection/evaluation/audit.py tests/unit/evaluation/test_audit.py
uv run ruff check --fix src/smoke_detection/evaluation/audit.py tests/unit/evaluation/test_audit.py
git add src/smoke_detection/evaluation/audit.py tests/unit/evaluation/test_audit.py
git commit -m "feat(evaluation): add validate_against_yaml + validate_all

Compares rendered string values from the audit table against the YAML
config. Skips gate rows. validate_all is the end-to-end check used by
the audit unit test (Spec 1) and the future Phase 2 CI gate.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: `figures_callback.py` — `out_dir_override` + test-time accumulators

**Files:**
- Modify: `src/smoke_detection/training/figures_callback.py`

- [ ] **Step 1: Modify `figures_callback.py` to add the override and accumulators**

Open `src/smoke_detection/training/figures_callback.py` and replace the `__init__` and `_resolve_out_dir` (currently a free function) per below. Find the `class TrainingFiguresCallback(Callback):` block and `def _resolve_out_dir(trainer: L.Trainer) -> Path:` at the bottom.

Replace `class TrainingFiguresCallback`'s `__init__` with:

```python
    def __init__(
        self,
        num_val_samples: int = 9,
        out_dir_override: Path | None = None,
    ):
        super().__init__()
        self.num_val_samples = num_val_samples
        self._out_dir_override = out_dir_override
        self.history: dict[str, list[float]] = {}
        # Test-time per-batch accumulators (populated by on_test_batch_end in Task 11/12).
        self._test_scores: list[float] = []
        self._test_labels: list[int] = []
        self._test_tp: int = 0
        self._test_tn: int = 0
        self._test_fp: int = 0
        self._test_fn: int = 0
        self._test_ious: list[float] = []
        self._test_area_ratios: list[float] = []
```

Then replace the standalone `def _resolve_out_dir(trainer: L.Trainer) -> Path:` function at the bottom of the file with a method on the class. Find the existing `on_train_end` method:

```python
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        out_dir = _resolve_out_dir(trainer)
```

Change `_resolve_out_dir(trainer)` to `self._resolve_out_dir(trainer)`.

Then add this method to the class (before `_plot_training_curves`):

```python
    def _resolve_out_dir(self, trainer: L.Trainer) -> Path:
        """Output dir: override > logger.log_dir/figures > default_root_dir/figures."""
        if self._out_dir_override is not None:
            return self._out_dir_override
        logger = trainer.logger
        if logger is not None and getattr(logger, "log_dir", None):
            base = Path(logger.log_dir)
        else:
            base = Path(trainer.default_root_dir or ".")
        return base / "figures"
```

Finally, delete the standalone `_resolve_out_dir(trainer: L.Trainer) -> Path:` function near the bottom of the file (the one that's no longer called).

- [ ] **Step 2: Run existing figures_callback-affecting tests to verify nothing broke**

```bash
uv run pytest tests/integration/test_module_consumes_batch.py tests/e2e/test_train_eval_cycle.py tests/e2e/test_fast_dev_run.py -v -m "e2e or not slow"
```

Expected: existing tests still pass (the new private fields don't affect any existing behavior; `_resolve_out_dir` now lives on the class but produces identical paths when `out_dir_override=None`).

- [ ] **Step 3: Commit**

```bash
uv run ruff format src/smoke_detection/training/figures_callback.py
uv run ruff check --fix src/smoke_detection/training/figures_callback.py
git add src/smoke_detection/training/figures_callback.py
git commit -m "refactor(training): figures_callback gains out_dir_override + test accumulators

Plumbs an override path so cli/eval.py can pin the legacy
<output>/<exp>/eval/ output location while the callback is reused for
test-time figures. Also moves _resolve_out_dir from a free function
onto the class so the override is applied uniformly.

Adds the per-batch accumulator fields the on_test_batch_end /
on_test_end hooks (Tasks 11/12) will populate. No behavior change yet.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: `figures_callback.py` — `on_test_batch_end` (classification + segmentation)

**Files:**
- Modify: `src/smoke_detection/training/figures_callback.py`
- Test: `tests/unit/training/test_figures_callback.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/training/test_figures_callback.py`:

```python
"""Unit tests for TrainingFiguresCallback (test-time hooks)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from lightning import Trainer

from smoke_detection.data.classification_datamodule import ClassificationDataModule
from smoke_detection.data.segmentation_datamodule import SegmentationDataModule
from smoke_detection.training.classification_module import ClassificationModule
from smoke_detection.training.figures_callback import TrainingFiguresCallback
from smoke_detection.training.segmentation_module import SegmentationModule


def test_callback_init_with_override():
    cb = TrainingFiguresCallback(out_dir_override=Path("/tmp/foo"))
    assert cb._out_dir_override == Path("/tmp/foo")
    assert cb._test_tp == 0 and cb._test_fp == 0
    assert cb._test_ious == []


@pytest.mark.e2e
def test_on_test_classification_writes_expected_pngs(synthetic_dataset_root, tmp_path):
    out = tmp_path / "eval"
    cb = TrainingFiguresCallback(out_dir_override=out)
    dm = ClassificationDataModule(
        data_root=synthetic_dataset_root, batch_size=2, num_workers=0,
        crop_size=90, balance="none",
    )
    module = ClassificationModule(in_channels=4, pretrained=False, lr=1e-3)
    trainer = Trainer(
        accelerator="cpu", devices=1, logger=False,
        callbacks=[cb], enable_checkpointing=False, max_epochs=1, fast_dev_run=True,
    )
    trainer.test(module, datamodule=dm)
    assert (out / "confusion_matrix.png").exists()
    assert (out / "roc_curve.png").exists()


@pytest.mark.e2e
def test_on_test_segmentation_writes_expected_pngs(synthetic_dataset_root, tmp_path):
    out = tmp_path / "eval"
    cb = TrainingFiguresCallback(out_dir_override=out)
    dm = SegmentationDataModule(
        data_root=synthetic_dataset_root, batch_size=1, num_workers=0, crop_size=90,
    )
    module = SegmentationModule(in_channels=4, n_classes=1, bilinear=True, lr=1e-3)
    trainer = Trainer(
        accelerator="cpu", devices=1, logger=False,
        callbacks=[cb], enable_checkpointing=False, max_epochs=1, fast_dev_run=True,
    )
    trainer.test(module, datamodule=dm)
    assert (out / "iou_distribution.png").exists()
    assert (out / "area_ratio_distribution.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/training/test_figures_callback.py -v -m "not slow and not gpu"
uv run pytest tests/unit/training/test_figures_callback.py -v -m "e2e"
```

Expected: `test_callback_init_with_override` passes; the two `e2e` tests fail because `on_test_batch_end` and `on_test_end` don't exist yet → no PNGs written.

- [ ] **Step 3: Add `on_test_batch_end` to `figures_callback.py`**

Append to `class TrainingFiguresCallback` (after `on_validation_epoch_end`):

```python
    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Per-batch accumulation for test-time figures.

        Mirrors the per-batch logic that lived inline in cli/eval.py before
        the callback consolidation. Runs in eval (no-grad) mode; the LightningModule
        already manages eval mode via Trainer.test.
        """
        with torch.no_grad():
            if isinstance(pl_module, ClassificationModule):
                logits = pl_module(batch["img"]).cpu().squeeze(1)
                probs = torch.sigmoid(logits).tolist()
                ys = batch["lbl"].int().tolist()
                self._test_scores.extend(probs)
                self._test_labels.extend(ys)
                for p, y in zip(probs, ys, strict=False):
                    pred = 1 if p >= 0.5 else 0
                    if pred == 1 and y == 1:
                        self._test_tp += 1
                    elif pred == 0 and y == 0:
                        self._test_tn += 1
                    elif pred == 1 and y == 0:
                        self._test_fp += 1
                    else:
                        self._test_fn += 1
            elif isinstance(pl_module, SegmentationModule):
                y = batch["fpt"].float().unsqueeze(1).to(pl_module.device)
                logits = pl_module(batch["img"].to(pl_module.device))
                preds = (logits >= 0).float()
                inter = (preds * y).sum(dim=(1, 2, 3))
                union = ((preds + y) > 0).float().sum(dim=(1, 2, 3))
                for k in range(y.shape[0]):
                    if y[k].sum() > 0 and preds[k].sum() > 0:
                        self._test_ious.append(float(inter[k] / union[k]))
                    a_pred = float(preds[k].sum())
                    a_true = float(y[k].sum())
                    if a_pred == 0 and a_true == 0:
                        self._test_area_ratios.append(1.0)
                    elif a_true == 0:
                        self._test_area_ratios.append(0.0)
                    else:
                        self._test_area_ratios.append(a_pred / a_true)
```

You'll need `from typing import Any` at the top of the file if not already present.

- [ ] **Step 4: Run tests (the e2e ones still won't pass because `on_test_end` is missing)**

```bash
uv run pytest tests/unit/training/test_figures_callback.py -v -m "e2e"
```

Expected: still failing — accumulators populate but no PNGs because `on_test_end` is missing.

- [ ] **Step 5: Commit (intermediate; on_test_end follows in next task)**

```bash
uv run ruff format src/smoke_detection/training/figures_callback.py tests/unit/training/test_figures_callback.py
uv run ruff check --fix src/smoke_detection/training/figures_callback.py tests/unit/training/test_figures_callback.py
git add src/smoke_detection/training/figures_callback.py tests/unit/training/test_figures_callback.py
git commit -m "feat(training): figures_callback accumulates test-time per-batch metrics

on_test_batch_end mirrors the inline per-batch math from cli/eval.py
(no semantic change). Accumulator fields populated; on_test_end hook
(next commit) consumes them.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: `figures_callback.py` — `on_test_end` + `extra_test_metrics`

**Files:**
- Modify: `src/smoke_detection/training/figures_callback.py`
- Modify: `tests/unit/training/test_figures_callback.py`

- [ ] **Step 1: Add failing test for `extra_test_metrics`**

Append to `tests/unit/training/test_figures_callback.py`:

```python
@pytest.mark.e2e
def test_extra_test_metrics_returns_area_ratio_after_segmentation(
    synthetic_dataset_root, tmp_path,
):
    cb = TrainingFiguresCallback(out_dir_override=tmp_path / "eval")
    dm = SegmentationDataModule(
        data_root=synthetic_dataset_root, batch_size=1, num_workers=0, crop_size=90,
    )
    module = SegmentationModule(in_channels=4, n_classes=1, bilinear=True, lr=1e-3)
    trainer = Trainer(
        accelerator="cpu", devices=1, logger=False,
        callbacks=[cb], enable_checkpointing=False, max_epochs=1, fast_dev_run=True,
    )
    trainer.test(module, datamodule=dm)
    extras = cb.extra_test_metrics()
    assert "mean_abs_area_ratio_error" in extras
    assert isinstance(extras["mean_abs_area_ratio_error"], float)
    assert extras["mean_abs_area_ratio_error"] >= 0.0


def test_extra_test_metrics_empty_before_test():
    """Before any test pass runs, extra_test_metrics returns {}."""
    cb = TrainingFiguresCallback()
    assert cb.extra_test_metrics() == {}


@pytest.mark.e2e
def test_extra_test_metrics_empty_for_classification(synthetic_dataset_root, tmp_path):
    cb = TrainingFiguresCallback(out_dir_override=tmp_path / "eval")
    dm = ClassificationDataModule(
        data_root=synthetic_dataset_root, batch_size=2, num_workers=0,
        crop_size=90, balance="none",
    )
    module = ClassificationModule(in_channels=4, pretrained=False, lr=1e-3)
    trainer = Trainer(
        accelerator="cpu", devices=1, logger=False,
        callbacks=[cb], enable_checkpointing=False, max_epochs=1, fast_dev_run=True,
    )
    trainer.test(module, datamodule=dm)
    # Classification has no derived test metrics beyond what Trainer.test returns.
    assert cb.extra_test_metrics() == {}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/training/test_figures_callback.py -v -m "not slow and not gpu"
uv run pytest tests/unit/training/test_figures_callback.py -v -m "e2e"
```

Expected: e2e tests still fail (PNGs not written + extra_test_metrics not defined).

- [ ] **Step 3: Add `on_test_end` and `extra_test_metrics` to the callback**

Append to `class TrainingFiguresCallback`:

```python
    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        from smoke_detection.evaluation.classification_metrics import (
            plot_confusion_matrix, plot_roc_curve,
        )
        from smoke_detection.evaluation.segmentation_metrics import (
            plot_area_ratio_distribution, plot_iou_distribution,
        )

        out_dir = self._resolve_out_dir(trainer)
        out_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(pl_module, ClassificationModule):
            try:
                plot_confusion_matrix(
                    self._test_tp, self._test_tn, self._test_fp, self._test_fn,
                    out_dir / "confusion_matrix.png",
                )
            except Exception as exc:
                log.warning("Failed to write confusion_matrix.png: %s", exc)
            try:
                plot_roc_curve(self._test_scores, self._test_labels,
                               out_dir / "roc_curve.png")
            except Exception as exc:
                log.warning("Failed to write roc_curve.png: %s", exc)
        elif isinstance(pl_module, SegmentationModule):
            if not self._test_ious and not self._test_area_ratios:
                log.info("Skipping seg test figures: no test samples accumulated")
                return
            try:
                plot_iou_distribution(self._test_ious, out_dir / "iou_distribution.png")
            except Exception as exc:
                log.warning("Failed to write iou_distribution.png: %s", exc)
            try:
                plot_area_ratio_distribution(self._test_area_ratios,
                                             out_dir / "area_ratio_distribution.png")
            except Exception as exc:
                log.warning("Failed to write area_ratio_distribution.png: %s", exc)

    def extra_test_metrics(self) -> dict[str, float]:
        """Return derived test-time metrics not produced by Trainer.test return value.

        Currently: mean_abs_area_ratio_error (segmentation only). Returns {} if
        no test pass has run, or if the module type doesn't produce extras.
        """
        if not self._test_area_ratios:
            return {}
        return {
            "mean_abs_area_ratio_error": float(
                np.mean([abs(1.0 - r) for r in self._test_area_ratios])
            ),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/training/test_figures_callback.py -v -m "not slow and not gpu"
uv run pytest tests/unit/training/test_figures_callback.py -v -m "e2e"
```

Expected: all 6 tests pass (3 unit + 3 e2e).

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/smoke_detection/training/figures_callback.py tests/unit/training/test_figures_callback.py
uv run ruff check --fix src/smoke_detection/training/figures_callback.py tests/unit/training/test_figures_callback.py
git add src/smoke_detection/training/figures_callback.py tests/unit/training/test_figures_callback.py
git commit -m "feat(training): figures_callback emits test-time figures + extra metrics

on_test_end produces confusion_matrix.png + roc_curve.png (cls) or
iou_distribution.png + area_ratio_distribution.png (seg). All matplotlib
errors swallowed-and-logged so plotting failures don't abort eval runs.

extra_test_metrics() exposes mean_abs_area_ratio_error to the
orchestrator without crossing private-attribute boundaries.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: `cli/eval.py` — thin to callback-driven

**Files:**
- Modify: `src/smoke_detection/cli/eval.py`

- [ ] **Step 1: Replace the entire content of `cli/eval.py`**

```python
"""CLI entry point for evaluation. Runs ``trainer.test`` with the figures
callback attached; the callback owns figure generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from lightning import Trainer

from smoke_detection.common.logging import get_logger
from smoke_detection.common.seed import seed_everything
from smoke_detection.configs.classification import ClassificationConfig
from smoke_detection.configs.loader import load_config
from smoke_detection.configs.segmentation import SegmentationConfig
from smoke_detection.data.classification_datamodule import ClassificationDataModule
from smoke_detection.data.segmentation_datamodule import SegmentationDataModule
from smoke_detection.training.classification_module import ClassificationModule
from smoke_detection.training.figures_callback import TrainingFiguresCallback
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
    module = ClassificationModule.load_from_checkpoint(str(ckpt), weights_only=False)
    figures_cb = TrainingFiguresCallback(out_dir_override=out_dir)
    trainer = Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[figures_cb],
        logger=False,
        enable_checkpointing=False,
    )
    trainer.test(module, datamodule=dm)
    log.info("Wrote classification eval figures to %s", out_dir)


def _eval_segmentation(cfg: SegmentationConfig, ckpt: Path, out_dir: Path) -> None:
    dm = SegmentationDataModule(
        data_root=cfg.paths.data_root,
        batch_size=1,
        num_workers=cfg.data.num_workers,
        crop_size=cfg.data.crop_size,
    )
    module = SegmentationModule.load_from_checkpoint(str(ckpt), weights_only=False)
    figures_cb = TrainingFiguresCallback(out_dir_override=out_dir)
    trainer = Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[figures_cb],
        logger=False,
        enable_checkpointing=False,
    )
    trainer.test(module, datamodule=dm)
    log.info("Wrote segmentation eval figures to %s", out_dir)


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

- [ ] **Step 2: Run existing eval tests + train/eval cycle**

```bash
uv run pytest tests/e2e/test_train_eval_cycle.py -v -m e2e
```

Expected: existing eval-cycle test passes (the CLI signature is unchanged; the `out_dir` default is unchanged; the figure outputs land in the same place).

- [ ] **Step 3: Commit**

```bash
uv run ruff format src/smoke_detection/cli/eval.py
uv run ruff check --fix src/smoke_detection/cli/eval.py
git add src/smoke_detection/cli/eval.py
git commit -m "refactor(cli): thin eval.py — figures callback owns plotting

Removes ~60 lines of inline per-batch + plotting code. cli/eval.py
now just builds datamodule + module + Trainer (with the figures
callback attached) + calls trainer.test. Output paths preserved via
the callback's out_dir_override.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: `scripts/report_parity.py` — thin to CLI wrapper over the library

**Files:**
- Modify: `scripts/report_parity.py`

- [ ] **Step 1: Replace `scripts/report_parity.py` with a CLI wrapper over `evaluation.parity`**

```python
"""CLI wrapper: load a checkpoint, run trainer.test, hand metrics to
evaluation.parity for the JSON report + stdout table.

This script is the smaller sibling of scripts/run_paper_parity.py:
- run_paper_parity.py drives train+test+report end-to-end
- report_parity.py only does the test+report half (assumes a checkpoint)

Usage:
    python scripts/report_parity.py \\
        --config configs/classification/paper.yaml \\
        --ckpt lightning_logs/.../checkpoints/last.ckpt \\
        [--out PATH/parity.json]

Exits 0 if overall.pass else 2; 1 on unexpected error.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from lightning import Trainer

from smoke_detection.common.seed import seed_everything
from smoke_detection.configs.classification import ClassificationConfig
from smoke_detection.configs.loader import load_config
from smoke_detection.configs.segmentation import SegmentationConfig
from smoke_detection.data.classification_datamodule import ClassificationDataModule
from smoke_detection.data.segmentation_datamodule import SegmentationDataModule
from smoke_detection.evaluation.parity import (
    format_table,
    translate_lightning_metrics,
    write_parity_report,
)
from smoke_detection.training.classification_module import ClassificationModule
from smoke_detection.training.figures_callback import TrainingFiguresCallback
from smoke_detection.training.segmentation_module import SegmentationModule


def _count_dataset_files(data_root: Path, task: str) -> dict[str, int]:
    """Walk <data_root>/<task>/{train,val,test} and count .tif files."""
    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    base = Path(data_root) / task
    for split in counts:
        split_dir = base / split
        if not split_dir.exists():
            continue
        for _root, _dirs, files in os.walk(split_dir):
            counts[split] += sum(1 for f in files if f.endswith(".tif"))
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report paper-parity metrics for a checkpoint")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output parity.json path (defaults to <ckpt-parent>/../parity.json)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.override)
    seed_everything(cfg.seed, deterministic=cfg.trainer.deterministic)

    if not args.ckpt.exists():
        print(f"ERROR: checkpoint not found: {args.ckpt}")
        return 1

    out = args.out or args.ckpt.parent.parent / "parity.json"

    if isinstance(cfg, ClassificationConfig):
        task = "classification"
        dm = ClassificationDataModule(
            data_root=cfg.paths.data_root,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            crop_size=cfg.data.crop_size,
            balance="none",
        )
        module = ClassificationModule.load_from_checkpoint(str(args.ckpt), weights_only=False)
    elif isinstance(cfg, SegmentationConfig):
        task = "segmentation"
        dm = SegmentationDataModule(
            data_root=cfg.paths.data_root,
            batch_size=1,
            num_workers=cfg.data.num_workers,
            crop_size=cfg.data.crop_size,
        )
        module = SegmentationModule.load_from_checkpoint(str(args.ckpt), weights_only=False)
    else:
        raise RuntimeError(f"Unsupported config type: {type(cfg).__name__}")

    figures_cb = TrainingFiguresCallback(out_dir_override=out.parent / "eval")
    trainer = Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[figures_cb],
        logger=False,
        enable_checkpointing=False,
    )
    results = trainer.test(module, datamodule=dm, verbose=False)
    raw = results[0] if results else {}
    canonical = translate_lightning_metrics(raw)
    canonical.update(figures_cb.extra_test_metrics())

    file_counts = _count_dataset_files(cfg.paths.data_root, task)
    report = write_parity_report(
        task=task,
        observed=canonical,
        config_path=args.config,
        checkpoint_path=args.ckpt,
        dataset_root=cfg.paths.data_root,
        dataset_file_count=file_counts,
        out_path=out,
    )
    print(format_table(report))
    print(f"\nWrote parity report: {out}")
    return 0 if report.overall.pass_ else 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify the script imports cleanly**

```bash
uv run python -c "import scripts.report_parity"
```

Expected: no error.

- [ ] **Step 3: Commit**

```bash
uv run ruff format scripts/report_parity.py
uv run ruff check --fix scripts/report_parity.py
git add scripts/report_parity.py
git commit -m "refactor(scripts): thin report_parity.py to a CLI wrapper

Reuses evaluation.parity for schema, thresholds, and JSON writing.
Reuses figures_callback for figures and extra_test_metrics. Same
behavior as before the library extraction; same exit code semantics
(0 on PASS, 2 on FAIL, 1 on error).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: `scripts/run_paper_parity.py` — orchestrator (with integration test)

**Files:**
- Create: `scripts/run_paper_parity.py`
- Test: `tests/integration/test_run_paper_parity.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/integration/test_run_paper_parity.py`:

```python
"""Integration test for scripts/run_paper_parity.py.

Uses the synthetic-dataset fixture + fast_dev_run + permissive
--paper-overrides so the orchestrator can exit 0 against random data.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_test_config(tmp_path: Path, dataset_root: Path, task: str) -> Path:
    """Build a minimal classification or segmentation YAML pointed at synthetic data."""
    if task == "classification":
        cfg = {
            "task": "classification", "seed": 1234,
            "trainer": {"max_epochs": 1, "accelerator": "cpu", "devices": 1,
                        "precision": "32", "deterministic": True,
                        "log_every_n_steps": 1, "fast_dev_run": True},
            "paths": {"data_root": str(dataset_root),
                      "output_dir": str(tmp_path / "logs"),
                      "experiment_name": "synthetic_cls"},
            "optim": {"lr": 1e-3, "momentum": 0.9, "weight_decay": 0.0, "scheduler": "none"},
            "model": {"backbone": "resnet50", "pretrained": False, "in_channels": 4},
            "data": {"batch_size": 2, "num_workers": 0, "crop_size": 90, "balance": "none"},
        }
    else:
        cfg = {
            "task": "segmentation", "seed": 1234,
            "trainer": {"max_epochs": 1, "accelerator": "cpu", "devices": 1,
                        "precision": "32", "deterministic": True,
                        "log_every_n_steps": 1, "fast_dev_run": True},
            "paths": {"data_root": str(dataset_root),
                      "output_dir": str(tmp_path / "logs"),
                      "experiment_name": "synthetic_seg"},
            "optim": {"lr": 1e-3, "momentum": 0.9, "weight_decay": 0.0, "scheduler": "none"},
            "model": {"architecture": "unet", "in_channels": 4, "n_classes": 1, "bilinear": True},
            "data": {"batch_size": 1, "num_workers": 0, "crop_size": 90},
        }
    yml = tmp_path / f"{task}.yaml"
    yml.write_text(yaml.safe_dump(cfg))
    return yml


def _write_permissive_overrides(tmp_path: Path) -> Path:
    """PAPER overrides that always pass (used so synthetic data can clear the gate)."""
    overrides = {
        "PAPER": {
            "classification": {"test_accuracy": 0.0, "test_auc": None},
            "segmentation": {"test_iou": 0.0, "test_img_accuracy": 0.0,
                             "mean_abs_area_ratio_error": 1.0},
        },
        "TOLERANCE": {
            "classification": {"test_accuracy": 1.0, "test_auc": None},
            "segmentation": {"test_iou": 1.0, "test_img_accuracy": 1.0,
                             "mean_abs_area_ratio_error": 1.0},
        },
    }
    p = tmp_path / "overrides.json"
    p.write_text(json.dumps(overrides))
    return p


@pytest.mark.e2e
def test_run_paper_parity_classification_smoke(synthetic_dataset_root, tmp_path):
    cfg_yaml = _write_test_config(tmp_path, synthetic_dataset_root, "classification")
    overrides = _write_permissive_overrides(tmp_path)
    proc = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "run_paper_parity.py"),
            "--task", "classification",
            "--config", str(cfg_yaml),
            "--paper-overrides", str(overrides),
        ],
        cwd=REPO_ROOT,
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    # Exit 0 means parity.json was written and overall.pass=True
    assert "OVERALL: PASS" in proc.stdout
    parity_path = next((tmp_path / "logs" / "synthetic_cls").rglob("parity.json"), None)
    assert parity_path is not None and parity_path.exists()
    blob = json.loads(parity_path.read_text())
    assert blob["task"] == "classification"
    assert blob["overall"]["pass"] is True


@pytest.mark.e2e
def test_run_paper_parity_segmentation_smoke(synthetic_dataset_root, tmp_path):
    cfg_yaml = _write_test_config(tmp_path, synthetic_dataset_root, "segmentation")
    overrides = _write_permissive_overrides(tmp_path)
    proc = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "run_paper_parity.py"),
            "--task", "segmentation",
            "--config", str(cfg_yaml),
            "--paper-overrides", str(overrides),
        ],
        cwd=REPO_ROOT,
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "OVERALL: PASS" in proc.stdout
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/integration/test_run_paper_parity.py -v -m e2e
```

Expected: failure — the script doesn't exist.

- [ ] **Step 3: Create `scripts/run_paper_parity.py`**

```python
"""End-to-end orchestrator: train (subprocess) → test (in-process with figures
callback) → parity report (in-process). One task per invocation.

Usage:
    python scripts/run_paper_parity.py --task classification \\
        --config configs/classification/paper.yaml

Optional:
    --skip-train             use existing checkpoint at <output>/<exp>/version_N/checkpoints/last.ckpt
    --version N              force log version
    --strict-paper           refuse to start unless paper.yaml validates clean
                             against PAPER_PARITY_AUDIT.md
    --paper-overrides PATH   test-only; JSON overriding PAPER+TOLERANCE
                             (used by integration tests; do not use in real parity runs)

Exits 0 on PASS, 2 on FAIL, 1 on unexpected error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from lightning import Trainer

from smoke_detection.common.logging import get_logger
from smoke_detection.common.seed import seed_everything
from smoke_detection.configs.classification import ClassificationConfig
from smoke_detection.configs.loader import load_config
from smoke_detection.configs.segmentation import SegmentationConfig
from smoke_detection.data.classification_datamodule import ClassificationDataModule
from smoke_detection.data.segmentation_datamodule import SegmentationDataModule
from smoke_detection.evaluation import parity as parity_lib
from smoke_detection.evaluation.audit import load_audit_table, validate_against_yaml
from smoke_detection.evaluation.parity import (
    format_table,
    translate_lightning_metrics,
    write_parity_report,
)
from smoke_detection.training.classification_module import ClassificationModule
from smoke_detection.training.figures_callback import TrainingFiguresCallback
from smoke_detection.training.segmentation_module import SegmentationModule

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _apply_paper_overrides(path: Path) -> None:
    """Test-only: hot-patch PAPER + TOLERANCE in evaluation.parity."""
    overrides = json.loads(path.read_text())
    if "PAPER" in overrides:
        for task, table in overrides["PAPER"].items():
            parity_lib.PAPER[task] = dict(table)
    if "TOLERANCE" in overrides:
        for task, table in overrides["TOLERANCE"].items():
            parity_lib.TOLERANCE[task] = dict(table)


def _resolve_version_dir(output_dir: Path, experiment_name: str, version: int | None) -> Path:
    """Return <output_dir>/<experiment_name>/version_N/. Pick latest if version=None,
    or version_0 if no versions exist yet."""
    base = output_dir / experiment_name
    if version is not None:
        return base / f"version_{version}"
    if not base.exists():
        return base / "version_0"
    existing = []
    for child in base.iterdir():
        m = re.match(r"version_(\d+)$", child.name)
        if m:
            existing.append(int(m.group(1)))
    if not existing:
        return base / "version_0"
    return base / f"version_{max(existing)}"


def _count_dataset_files(data_root: Path, task: str) -> dict[str, int]:
    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    base = Path(data_root) / task
    for split in counts:
        split_dir = base / split
        if not split_dir.exists():
            continue
        for _root, _dirs, files in os.walk(split_dir):
            counts[split] += sum(1 for f in files if f.endswith(".tif"))
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train + test + report paper-parity metrics for one task"
    )
    parser.add_argument("--task", required=True, choices=["classification", "segmentation"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--skip-train", action="store_true",
                        help="Use existing checkpoint instead of training fresh")
    parser.add_argument("--version", type=int, default=None,
                        help="Force log version_N; default picks latest existing or 0")
    parser.add_argument("--strict-paper", action="store_true",
                        help="Validate paper.yaml against PAPER_PARITY_AUDIT.md before training")
    parser.add_argument("--paper-overrides", type=Path, default=None,
                        help="TEST ONLY: JSON overriding PAPER+TOLERANCE constants")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args(argv)

    if args.paper_overrides is not None:
        _apply_paper_overrides(args.paper_overrides)

    cfg = load_config(args.config, overrides=args.override)
    seed_everything(cfg.seed, deterministic=cfg.trainer.deterministic)

    if cfg.task != args.task:
        print(f"ERROR: --task {args.task} doesn't match config task {cfg.task}", file=sys.stderr)
        return 1

    if args.strict_paper:
        try:
            tables = load_audit_table()
        except Exception as exc:
            print(f"ERROR: failed to load audit table: {exc}", file=sys.stderr)
            return 1
        discrepancies = validate_against_yaml(tables[args.task], args.config)
        if discrepancies:
            print("ERROR: --strict-paper found audit/YAML discrepancies:", file=sys.stderr)
            for d in discrepancies:
                print(f"  - {d}", file=sys.stderr)
            return 1

    version_dir = _resolve_version_dir(cfg.paths.output_dir, cfg.paths.experiment_name, args.version)
    version_dir.mkdir(parents=True, exist_ok=True)

    # Phase A: TRAIN
    if not args.skip_train:
        train_cmd = [
            sys.executable, "-m", "smoke_detection.cli.train",
            "--config", str(args.config),
        ]
        for ov in args.override:
            train_cmd.extend(["--override", ov])
        log.info("TRAIN: %s", " ".join(train_cmd))
        proc = subprocess.run(train_cmd, cwd=REPO_ROOT)
        if proc.returncode != 0:
            print(f"ERROR: training subprocess exited {proc.returncode}", file=sys.stderr)
            return 1
        # Pick the latest version_N created by training (in case of fresh experiment_name)
        version_dir = _resolve_version_dir(cfg.paths.output_dir, cfg.paths.experiment_name, args.version)

    # Locate checkpoint
    ckpt = version_dir / "checkpoints" / "last.ckpt"
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        return 1

    # Phase B: TEST + REPORT (in-process, single pass)
    if args.task == "classification":
        cfg_typed = cfg
        assert isinstance(cfg_typed, ClassificationConfig)
        dm = ClassificationDataModule(
            data_root=cfg_typed.paths.data_root,
            batch_size=cfg_typed.data.batch_size,
            num_workers=cfg_typed.data.num_workers,
            crop_size=cfg_typed.data.crop_size,
            balance="none",
        )
        module = ClassificationModule.load_from_checkpoint(str(ckpt), weights_only=False)
    else:
        cfg_typed = cfg
        assert isinstance(cfg_typed, SegmentationConfig)
        dm = SegmentationDataModule(
            data_root=cfg_typed.paths.data_root,
            batch_size=1,
            num_workers=cfg_typed.data.num_workers,
            crop_size=cfg_typed.data.crop_size,
        )
        module = SegmentationModule.load_from_checkpoint(str(ckpt), weights_only=False)

    figures_cb = TrainingFiguresCallback(out_dir_override=version_dir / "eval")
    trainer = Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[figures_cb],
        logger=False,
        enable_checkpointing=False,
    )
    results = trainer.test(module, datamodule=dm, verbose=False)
    raw = results[0] if results else {}
    canonical = translate_lightning_metrics(raw)
    canonical.update(figures_cb.extra_test_metrics())

    file_counts = _count_dataset_files(cfg.paths.data_root, args.task)
    out_path = version_dir / "parity.json"
    report = write_parity_report(
        task=args.task,
        observed=canonical,
        config_path=args.config,
        checkpoint_path=ckpt,
        dataset_root=cfg.paths.data_root,
        dataset_file_count=file_counts,
        out_path=out_path,
    )
    print(format_table(report))
    print(f"\nWrote parity report: {out_path}")
    return 0 if report.overall.pass_ else 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run integration tests**

```bash
uv run pytest tests/integration/test_run_paper_parity.py -v -m e2e
```

Expected: 2 passed (cls smoke + seg smoke).

If a test times out, check that `fast_dev_run: True` is set in the test config (it is) and that the synthetic dataset fixture is producing files (existing fixture; should be fine).

- [ ] **Step 5: Commit**

```bash
uv run ruff format scripts/run_paper_parity.py tests/integration/test_run_paper_parity.py
uv run ruff check --fix scripts/run_paper_parity.py tests/integration/test_run_paper_parity.py
git add scripts/run_paper_parity.py tests/integration/test_run_paper_parity.py
git commit -m "feat(scripts): add run_paper_parity.py orchestrator

Drives one task end-to-end: train (subprocess to cli/train.py) →
test (in-process Trainer + figures callback for both figures and
metrics in one pass) → parity JSON. Exits 0 on PASS, 2 on FAIL,
1 on unexpected error.

--strict-paper validates the audit doc agrees with paper.yaml before
spending GPU time. --paper-overrides is test-only; integration tests
use it so synthetic data can clear permissive thresholds.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: `Makefile` — `parity-cls`, `parity-seg`, `parity` targets

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Append to `Makefile`**

After the existing `eval-seg:` target, before `clean:`, insert:

```makefile

parity-cls:
	uv run python scripts/run_paper_parity.py --task classification \
	    --config configs/classification/paper.yaml

parity-seg:
	uv run python scripts/run_paper_parity.py --task segmentation \
	    --config configs/segmentation/paper.yaml

# Sequential. parity-seg only runs if parity-cls exits 0.
# Exit 2 from the orchestrator (parity FAIL) halts the chain — that's intended;
# RUNBOOK_paper_parity.md tells you what to do next.
parity: parity-cls parity-seg
```

Also update the `.PHONY:` line at the top to include the three new targets:

```makefile
.PHONY: install lint format test test-e2e test-gpu test-all clean train-cls train-seg eval-cls eval-seg parity-cls parity-seg parity
```

- [ ] **Step 2: Verify Make targets parse**

```bash
make -n parity-cls parity-seg parity 2>&1 | head -20
```

Expected: prints the three `uv run python ...` commands without executing them.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "feat(makefile): add parity-cls, parity-seg, parity targets

Wraps scripts/run_paper_parity.py for the GPU-box workflow. 'make parity'
runs both tasks sequentially; chain halts on FAIL (orchestrator exit 2).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 17: `notebooks/results.ipynb` — parameterized notebook

**Files:**
- Modify: `notebooks/results.ipynb`

- [ ] **Step 1: Replace `notebooks/results.ipynb` with the parameterized notebook**

Write the following to `notebooks/results.ipynb`. This is JSON in nbformat-4 shape; copy verbatim:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Smoke Plume Detection — Results\n",
    "\n",
    "Renders the paper-parity report and figures produced by `scripts/run_paper_parity.py`. Parameterized via papermill (or env vars). Set `SMOKEDET_DEMO=1` for a self-contained demo run against the synthetic dataset fixture.\n",
    "\n",
    "**Spec reference:** `docs/superpowers/specs/2026-04-25-spec-1-paper-parity-design.md`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {"tags": ["parameters"]},
   "outputs": [],
   "source": [
    "import os\n",
    "from pathlib import Path\n",
    "\n",
    "task = \"classification\"  # 'classification' or 'segmentation'\n",
    "parity_json_path = os.environ.get(\n",
    "    \"SMOKEDET_PARITY_JSON\",\n",
    "    \"lightning_logs/classification_4ch_resnet50_paper/version_0/parity.json\",\n",
    ")\n",
    "figures_dir = os.environ.get(\n",
    "    \"SMOKEDET_FIGURES_DIR\",\n",
    "    \"lightning_logs/classification_4ch_resnet50_paper/version_0/eval\",\n",
    ")\n",
    "checkpoint_path = os.environ.get(\n",
    "    \"SMOKEDET_CHECKPOINT_PATH\",\n",
    "    \"lightning_logs/classification_4ch_resnet50_paper/version_0/checkpoints/last.ckpt\",\n",
    ")\n",
    "demo_mode = os.environ.get(\"SMOKEDET_DEMO\", \"0\") == \"1\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Setup: imports + (if demo_mode) self-sufficient synthetic-fixture run\n",
    "import json\n",
    "import sys\n",
    "import tempfile\n",
    "from IPython.display import Image, Markdown, display\n",
    "\n",
    "def _find_repo_root() -> Path:\n",
    "    # Walk up from CWD looking for pyproject.toml; works whether the notebook\n",
    "    # is run from notebooks/ (jupyter) or repo-root (papermill/nbconvert).\n",
    "    p = Path.cwd().resolve()\n",
    "    for cand in [p, *p.parents]:\n",
    "        if (cand / 'pyproject.toml').exists():\n",
    "            return cand\n",
    "    return p\n",
    "\n",
    "if demo_mode:\n",
    "    REPO_ROOT = _find_repo_root()\n",
    "    sys.path.insert(0, str(REPO_ROOT))\n",
    "    from tests._data import build_synthetic_prepared_tree\n",
    "\n",
    "    demo_root = Path(tempfile.mkdtemp(prefix='smokedet_nb_'))\n",
    "    demo_data = demo_root / 'data'\n",
    "    build_synthetic_prepared_tree(demo_data)\n",
    "    os.environ['SMOKEDET_DATA_ROOT'] = str(demo_data)\n",
    "\n",
    "    import yaml as _yaml\n",
    "    cfg_yaml = demo_root / 'cls.yaml'\n",
    "    cfg_yaml.write_text(_yaml.safe_dump({\n",
    "        'task': 'classification', 'seed': 1234,\n",
    "        'trainer': {'max_epochs': 1, 'accelerator': 'cpu', 'devices': 1,\n",
    "                    'precision': '32', 'deterministic': True,\n",
    "                    'log_every_n_steps': 1, 'fast_dev_run': True},\n",
    "        'paths': {'data_root': str(demo_data),\n",
    "                  'output_dir': str(demo_root / 'logs'),\n",
    "                  'experiment_name': 'demo_cls'},\n",
    "        'optim': {'lr': 1e-3, 'momentum': 0.9, 'weight_decay': 0.0, 'scheduler': 'none'},\n",
    "        'model': {'backbone': 'resnet50', 'pretrained': False, 'in_channels': 4},\n",
    "        'data': {'batch_size': 2, 'num_workers': 0, 'crop_size': 90, 'balance': 'none'},\n",
    "    }))\n",
    "    overrides = demo_root / 'ov.json'\n",
    "    overrides.write_text(json.dumps({\n",
    "        'PAPER': {'classification': {'test_accuracy': 0.0, 'test_auc': None}},\n",
    "        'TOLERANCE': {'classification': {'test_accuracy': 1.0, 'test_auc': None}},\n",
    "    }))\n",
    "    import subprocess\n",
    "    subprocess.run([sys.executable,\n",
    "                    str(REPO_ROOT / 'scripts' / 'run_paper_parity.py'),\n",
    "                    '--task', 'classification', '--config', str(cfg_yaml),\n",
    "                    '--paper-overrides', str(overrides)],\n",
    "                   cwd=REPO_ROOT, check=True)\n",
    "    parity_json_path = str(next((demo_root / 'logs' / 'demo_cls').rglob('parity.json')))\n",
    "    figures_dir = str(Path(parity_json_path).parent / 'eval')\n",
    "    task = 'classification'\n",
    "\n",
    "parity_json_path = Path(parity_json_path)\n",
    "figures_dir = Path(figures_dir)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Provenance + parity summary\n",
    "if not parity_json_path.exists():\n",
    "    display(Markdown(\n",
    "        f'> **No parity.json found at `{parity_json_path}`.**\\n>\\n'\n",
    "        f'> Run `make parity-{task[:3]}` on the GPU box, then re-run this notebook.\\n>\\n'\n",
    "        f'> Or set `SMOKEDET_DEMO=1` for a self-contained demo run.'\n",
    "    ))\n",
    "else:\n",
    "    report = json.loads(parity_json_path.read_text())\n",
    "    prov = report['provenance']\n",
    "    overall = report['overall']\n",
    "    display(Markdown(f'## Provenance\\n\\n'\n",
    "        f'| field | value |\\n|---|---|\\n'\n",
    "        f'| task | {report[\"task\"]} |\\n'\n",
    "        f'| produced_at | {report[\"produced_at\"]} |\\n'\n",
    "        f'| git_commit | `{prov[\"git_commit\"]}` (dirty: {prov[\"git_dirty\"]}) |\\n'\n",
    "        f'| config | `{prov[\"config_path\"]}` (sha256: `{prov[\"config_sha256\"][:12]}...`) |\\n'\n",
    "        f'| checkpoint | `{prov[\"checkpoint_path\"]}` |\\n'\n",
    "        f'| dataset | `{prov[\"dataset_root\"]}` ({prov[\"dataset_file_count\"]}) |\\n\\n'\n",
    "        f'## Overall: **{\"PASS\" if overall[\"pass\"] else \"FAIL\"}**  '\n",
    "        f'(passed={overall[\"passed_count\"]} failed={overall[\"failed_count\"]} ungated={overall[\"ungated_count\"]})'\n",
    "    ))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Per-metric table\n",
    "if parity_json_path.exists():\n",
    "    report = json.loads(parity_json_path.read_text())\n",
    "    rows = ['| metric | value | paper | delta | tolerance | gate |',\n",
    "            '|---|---|---|---|---|---|']\n",
    "    for name, m in report['metrics'].items():\n",
    "        v = '—' if m['value'] is None else f\"{m['value']:.4f}\"\n",
    "        p = '—' if m['paper'] is None else f\"{m['paper']:.4f}\"\n",
    "        d = '—' if m['delta'] is None else f\"{m['delta']:+.4f}\"\n",
    "        t = '—' if m['tolerance'] is None else f\"{m['tolerance']:.4f}\"\n",
    "        if m['pass'] is None:\n",
    "            g = 'ungated'\n",
    "        else:\n",
    "            g = '✅ PASS' if m['pass'] else '❌ FAIL'\n",
    "        rows.append(f'| `{name}` | {v} | {p} | {d} | {t} | {g} |')\n",
    "    display(Markdown('\\n'.join(rows)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Embed eval figures\n",
    "if task == 'classification':\n",
    "    candidates = ['confusion_matrix.png', 'roc_curve.png']\n",
    "else:\n",
    "    candidates = ['iou_distribution.png', 'area_ratio_distribution.png']\n",
    "for fn in candidates:\n",
    "    p = figures_dir / fn\n",
    "    if p.exists():\n",
    "        display(Markdown(f'### `{fn}`'))\n",
    "        display(Image(filename=str(p)))\n",
    "    else:\n",
    "        display(Markdown(f'> `{fn}` not found at `{p}`'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Embed training figures (val_predictions = qualitative samples; training_curves = loss/acc/IoU curves)\n",
    "training_figures_dir = figures_dir.parent / 'figures'\n",
    "for fn in ('training_curves.png', 'val_predictions.png'):\n",
    "    p = training_figures_dir / fn\n",
    "    if p.exists():\n",
    "        display(Markdown(f'### `{fn}`'))\n",
    "        display(Image(filename=str(p)))\n",
    "    else:\n",
    "        display(Markdown(f'> `{fn}` not found at `{p}` (will appear after a full training run)'))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Capstone narrative\n",
    "\n",
    "_(Edit this section after the GPU-box parity run completes.)_\n",
    "\n",
    "**Parity status:** `<TBD: PASS / FAIL with X gap on metric Y>`\n",
    "\n",
    "**Notable findings:** `<TBD>`\n",
    "\n",
    "**Next step (Spec 2 / §1.2):** define generalization splits and the multi-stratum eval against this baseline."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Sanity-check the notebook is valid JSON / nbformat**

```bash
uv run python -c "
import json
with open('notebooks/results.ipynb') as f:
    nb = json.load(f)
assert nb['nbformat'] == 4
assert any('parameters' in c.get('metadata', {}).get('tags', []) for c in nb['cells']), 'no parameters cell'
print(f'cells: {len(nb[\"cells\"])}')
"
```

Expected: `cells: 8`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/results.ipynb
git commit -m "feat(notebooks): replace results.ipynb with parameterized version

Papermill-friendly parameters cell + env-var fallback + SMOKEDET_DEMO=1
self-contained mode that runs the synthetic-dataset fixture through
run_paper_parity.py. Renders parity report + provenance table + per-
metric pass/fail + eval/training figures. Capstone narrative section
left as TBD for the GPU-box re-run.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 18: Notebook smoke test

**Files:**
- Create: `tests/integration/test_notebook_smoke.py`

- [ ] **Step 1: Write the test**

Create `tests/integration/test_notebook_smoke.py`:

```python
"""Smoke test: notebooks/results.ipynb runs end-to-end in demo mode."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.e2e
def test_results_notebook_executes_in_demo_mode(tmp_path):
    """Run the notebook with SMOKEDET_DEMO=1; assert it finishes without error."""
    if shutil.which("jupyter") is None:
        pytest.skip("jupyter not on PATH (install via 'uv sync --extra notebooks')")
    nb_in = REPO_ROOT / "notebooks" / "results.ipynb"
    nb_out = tmp_path / "results_executed.ipynb"
    env = os.environ.copy()
    env["SMOKEDET_DEMO"] = "1"
    proc = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
            "--execute", str(nb_in),
            "--output", str(nb_out),
            "--ExecutePreprocessor.timeout=120",
        ],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert nb_out.exists()
```

- [ ] **Step 2: Run test (skipped if jupyter not installed)**

```bash
uv sync --extra notebooks
uv run pytest tests/integration/test_notebook_smoke.py -v -m e2e
```

Expected: 1 passed (or skipped if jupyter is unavailable; explicit skip is fine).

- [ ] **Step 3: Commit**

```bash
uv run ruff format tests/integration/test_notebook_smoke.py
uv run ruff check --fix tests/integration/test_notebook_smoke.py
git add tests/integration/test_notebook_smoke.py
git commit -m "test(notebook): smoke-test results.ipynb in SMOKEDET_DEMO=1 mode

Per-Spec-1 minimal CI-readiness check. Phase 2 §2.4 will replace this
with proper papermill + nbstripout enforcement.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 19: `docs/RUNBOOK_paper_parity.md` — GPU-box workflow

**Files:**
- Create: `docs/RUNBOOK_paper_parity.md`

- [ ] **Step 1: Write the runbook**

Create `docs/RUNBOOK_paper_parity.md`:

```markdown
# Runbook — Paper Parity (GPU box)

Sequential workflow for executing Spec 1 (`docs/superpowers/specs/2026-04-25-spec-1-paper-parity-design.md`) on a CUDA-capable machine. Read top to bottom.

**Decision rule on parity FAIL:** *one* retry with strict-paper hyperparameters; if still FAIL, append a "Known parity gap" section to `docs/PAPER_PARITY_AUDIT.md` and accept. Do not loop.

## 1. Pre-flight checklist

```bash
# CUDA available?
uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.cuda.get_device_name(0))"

# Dataset prepared?
ls data/classification/train/positive/*.tif | head
# If empty:
uv run python scripts/prepare_dataset.py \
    --source /path/to/4250706 \
    --output data
# (Use --mode hardlink to avoid 2x disk for the seg/cls duplicate copies, if your
# filesystem supports it.)

# Clean repo (provenance hates dirty trees)
git status

# Sync deps to lockfile
uv sync --extra dev
```

## 2. Single-command happy path

```bash
make parity   # runs parity-cls then parity-seg sequentially, ≈ 4–10 hrs total
```

If both exit 0, jump to step 5. Otherwise, see step 3 / 4.

## 3. If `parity-cls` exits 2 (FAIL)

Inspect which metric failed:

```bash
cat lightning_logs/classification_4ch_resnet50_paper/version_*/parity.json | tail -50
```

Re-run with the audit-doc strict check, in case a YAML drifted from the audit:

```bash
uv run python scripts/run_paper_parity.py --task classification \
    --config configs/classification/paper.yaml --strict-paper
```

- If `--strict-paper` exits 1 with discrepancies: fix the YAML or the audit doc, re-run.
- If `--strict-paper` ran clean and parity STILL fails: per Spec 1 question 3b, **document and accept**. Append to `docs/PAPER_PARITY_AUDIT.md`:

  ```markdown
  ## Known parity gap (classification, <date>)

  - **Metric:** test_accuracy = 0.927 vs. paper 0.943 (Δ −0.016, threshold ±0.020)
  - **Hypothesis:** 4-channel input vs. paper's 12-channel L1C variant (parent design rationale, open question 1)
  - **Decision:** accept as new baseline; tag `v0.3.0`; revisit in Spec 2 if generalization improvements depend on the design choice.
  ```

  Move on. Do not loop.

## 4. Same logic for `parity-seg`

`parity-seg` typically takes longer (300 epochs vs. 100). Consider running overnight:

```bash
nohup make parity-seg > parity-seg.log 2>&1 &
```

If FAIL: same decision rule as classification. Common gaps are slightly low IoU (< 0.578) — accept and document if `--strict-paper` runs clean.

## 5. Capture artifacts back to the repo

The `.ckpt` files stay on the GPU box; their sha256 lives in `parity.json`. Commit:

```bash
# Both parity.json files
git add lightning_logs/classification_4ch_resnet50_paper/version_*/parity.json
git add lightning_logs/segmentation_4ch_unet_paper/version_*/parity.json
# (You may need to relax .gitignore or use 'git add -f' if lightning_logs is gitignored;
# adjust per your repo policy.)

# Re-run notebook against the produced artifacts
uv sync --extra notebooks
uv run papermill notebooks/results.ipynb /tmp/cls_results.ipynb \
    -p task classification \
    -p parity_json_path lightning_logs/classification_4ch_resnet50_paper/version_0/parity.json \
    -p figures_dir lightning_logs/classification_4ch_resnet50_paper/version_0/eval

# Review /tmp/cls_results.ipynb visually, fill in the "Capstone narrative" markdown cell,
# then commit:
cp /tmp/cls_results.ipynb notebooks/results.ipynb   # or hand-edit to keep parameters cell
git add notebooks/results.ipynb

git add docs/PAPER_PARITY_AUDIT.md   # if a Known parity gap section was added
git commit -m "feat(spec-1): paper-parity baseline established"
```

## 6. Verify-against-paper cleanup pass (when PDF access is restored)

`docs/PAPER_PARITY_AUDIT.md` was authored without PDF access; every "could be in paper" row carries `verify-against-paper` in its notes column. When you regain PDF access:

1. Walk every `verify-against-paper` row.
2. If the paper documents the value, update `paper_value` and `source` columns; replace `verify-against-paper` in `notes` with a `paper §X.Y` citation.
3. If the paper omits the value, change the note to `paper silent` and leave `paper_value=?` and `source=hsg-aiml`.
4. Re-run the audit unit test:
   ```bash
   uv run pytest tests/unit/evaluation/test_audit.py -v
   ```
5. Commit.

## 7. Tag the release

```bash
git tag v0.3.0
git push origin v0.3.0   # ASK before pushing tags to a shared remote
```

CHANGELOG entry to add (manually):

```markdown
## v0.3.0 — Paper-parity baseline

- Established Mommert et al. 2020 paper-parity baseline (Spec 1)
- New `evaluation/parity.py` library + `scripts/run_paper_parity.py` orchestrator
- New `docs/PAPER_PARITY_AUDIT.md` per-knob audit table
- Figures callback consolidated across train + test
- See `docs/RUNBOOK_paper_parity.md` for reproduction
- Per-task parity status:
  - Classification: PASS / FAIL with documented gap (fill in)
  - Segmentation: PASS / FAIL with documented gap (fill in)
```

## 8. Failure escape hatches

- **OOM at paper batch size:** lower `data.batch_size` via override; document the deviation in `PAPER_PARITY_AUDIT.md` notes column; accept the parity gap as "non-paper batch size."
  ```bash
  uv run python scripts/run_paper_parity.py --task classification \
      --config configs/classification/paper.yaml \
      --override data.batch_size=16
  ```

- **Training diverges (NaN loss) at `lr=0.3`:** known risk with high LR + small batches. Add gradient clipping; document the deviation:
  ```bash
  uv run python scripts/run_paper_parity.py --task classification \
      --config configs/classification/paper.yaml \
      --override trainer.gradient_clip_val=1.0
  ```

- **`--strict-paper` keeps complaining about a value drift you don't understand:** check `evaluation.audit._render`'s rendering rules — booleans, None, floats render specifically. If `paper.yaml` has `lr: 0.3` and the audit doc has `0.30000`, that's a renderer mismatch — fix the audit doc to match `repr(0.3)`.
```

- [ ] **Step 2: Verify the runbook is valid markdown**

```bash
uv run python -c "
text = open('docs/RUNBOOK_paper_parity.md').read()
assert '## 1. Pre-flight checklist' in text
assert '## 7. Tag the release' in text
assert 'verify-against-paper' in text
print('OK')
"
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add docs/RUNBOOK_paper_parity.md
git commit -m "docs: add RUNBOOK_paper_parity.md (GPU-box workflow + iteration rule)

Sequential workflow for executing Spec 1 on a CUDA-capable machine.
Encodes the iteration decision rule (one retry with strict-paper, then
accept and document), the verify-against-paper cleanup pass for when
PDF access returns, and the failure escape hatches for OOM / NaN /
audit drift.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 20: Final verification — full test suite + lint pass

**Files:** none (verification only).

- [ ] **Step 1: Run the default test suite**

```bash
uv run pytest -v
```

Expected: all default-marker tests pass (excludes slow, gpu, e2e). New unit tests should be visible in the output.

- [ ] **Step 2: Run the e2e suite**

```bash
uv run pytest -v -m e2e --timeout=300
```

Expected: existing e2e tests + the new ones (figures callback `on_test_*`, run_paper_parity smoke ×2, notebook smoke) all pass.

- [ ] **Step 3: Lint pass**

```bash
uv run ruff check .
uv run ruff format --check .
```

Expected: no errors. If ruff finds anything, fix and re-run.

- [ ] **Step 4: Manual smoke — run the orchestrator against synthetic data on this box**

```bash
# This box is GPU-less; the synthetic fixture + fast_dev_run + permissive
# overrides path is what the integration test exercises. As a manual sanity check:
uv run pytest tests/integration/test_run_paper_parity.py::test_run_paper_parity_classification_smoke -v -m e2e -s
```

Expected: PASS, with the orchestrator's table printed via `-s`.

- [ ] **Step 5: Update CHANGELOG.md (placeholder for the GPU-box pass)**

Open `CHANGELOG.md` and add an `## [Unreleased]` section near the top (or use the existing one if present):

```markdown
## [Unreleased]

### Added — Spec 1 (Phase 1 §1.1) — code-only landing
- `evaluation/parity.py`: pure-Python parity report library (types, threshold logic, JSON writer, provenance hashing, Lightning key translation)
- `evaluation/audit.py`: PAPER_PARITY_AUDIT.md parser + YAML cross-validator
- `scripts/run_paper_parity.py`: end-to-end orchestrator (one task per invocation)
- `docs/PAPER_PARITY_AUDIT.md`: per-knob audit table (conservative initial state, pending PDF verification)
- `docs/RUNBOOK_paper_parity.md`: GPU-box workflow + iteration rule
- Makefile targets: `parity-cls`, `parity-seg`, `parity`
- `notebooks/results.ipynb`: parameterized (papermill + env-var) with SMOKEDET_DEMO=1 self-contained mode
- `figures_callback.TrainingFiguresCallback`: gains `on_test_batch_end`, `on_test_end`, `out_dir_override`, `extra_test_metrics()` — single source of truth for plotting across train + test

### Changed
- `cli/eval.py`: thinned to callback-driven (no inline plotting)
- `scripts/report_parity.py`: thinned to a CLI wrapper over `evaluation.parity`
- `configs/{classification,segmentation}/paper.yaml`: per-knob source attribution comments

### Pending (GPU box)
- Two `parity.json` artifacts (one per task)
- Updated `notebooks/results.ipynb` with populated narrative
- v0.3.0 tag
```

- [ ] **Step 6: Commit the CHANGELOG**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): record Spec 1 code-only landing on phase-1-generalization

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 7: Final status check**

```bash
git log --oneline phase-1-generalization ^master
```

Expected output (in order, one line per task commit):
```
<sha> docs(changelog): record Spec 1 code-only landing
<sha> docs: add RUNBOOK_paper_parity.md
<sha> test(notebook): smoke-test results.ipynb
<sha> feat(notebooks): replace results.ipynb with parameterized version
<sha> feat(makefile): add parity-cls, parity-seg, parity targets
<sha> feat(scripts): add run_paper_parity.py orchestrator
<sha> refactor(scripts): thin report_parity.py
<sha> refactor(cli): thin eval.py
<sha> feat(training): figures_callback emits test-time figures
<sha> feat(training): figures_callback accumulates test-time per-batch metrics
<sha> refactor(training): figures_callback gains out_dir_override
<sha> feat(evaluation): add validate_against_yaml + validate_all
<sha> feat(evaluation): add audit.py
<sha> docs: add PAPER_PARITY_AUDIT.md
<sha> chore(configs): annotate paper.yaml
<sha> feat(evaluation): add write_parity_report + format_table
<sha> feat(evaluation): translate Lightning metric keys
<sha> feat(evaluation): add provenance helpers
<sha> feat(evaluation): add evaluate_thresholds()
<sha> feat(evaluation): add parity types + threshold constants
<sha> docs: add Spec 1 design
```

The plan is complete on this box. The user takes the GPU box, runs through `docs/RUNBOOK_paper_parity.md`, captures the two `parity.json` files, edits the notebook narrative, and tags `v0.3.0`.
