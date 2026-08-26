# Spec 1 — Paper-Parity Baseline (Phase 1 §1.1)

**Status:** proposed 2026-04-25
**Branch:** `phase-1-generalization`
**Release tag at completion:** `v0.3.0`
**Parent roadmap:** [`docs/superpowers/specs/2026-04-25-todo-roadmap-design.md`](2026-04-25-todo-roadmap-design.md)

## Context

This is the first of three sequential specs implementing Phase 1 of `TODO.md`.
The full Phase 1 contains six sub-sections (1.1–1.6) that don't all belong in
one design conversation. The agreed decomposition:

1. **Spec 1 (this doc)** — §1.1 baseline (gating).
2. **Spec 2** — §1.2 + §1.6 measurement framework (geographic / seasonal /
   plume-type holdout splits, multi-stratum eval, ablation runner,
   `paper-robust*` configs, `GENERALIZATION_REPORT.md`). Designed *after*
   Spec 1's parity numbers come back, since those numbers may reframe what
   §1.2 needs to measure.
3. **Spec 3** — §1.3 + §1.4 + §1.5 improvements (augmentation upgrades,
   inference-time techniques, SWA + label smoothing). Designed last,
   consumes Spec 2's measurement framework.

This spec covers Spec 1 only. Spec 2 and Spec 3 are out of scope.

### What §1.1 demands

From `TODO.md` and the parent design rationale:

- Train classifier (ResNet-50) and segmenter (U-Net) on Zenodo to paper parity
- Extend `scripts/report_parity.py` to emit a signed JSON report
  (acc/AUC/IoU, deltas vs. Mommert 2020 Table 1, pass/fail thresholds)
- Verify `configs/{classification,segmentation}/paper.yaml` hyperparameters
  match the publication
- Wire `training/figures_callback.py` end-of-eval artifacts + regression
  test that expected files are written
- Populate `notebooks/results.ipynb` (confusion matrix, ROC, IoU +
  area-ratio histograms, per-image qualitative samples)

### Constraints

- **Split-machine workflow:** the spec and plan are written on a no-GPU
  device; execution happens on a separate GPU device. The plan must be
  fully executable on the GPU box without further design conversations.
- **Inference-target awareness (parent doc):** Phase 1.1 doesn't deploy
  anything; Orin AGX considerations are deferred to Specs 2/3 and Phase 3.
- **No PDF access right now:** the audit doc (see §4) is conservative
  about citations until the user regains access to the Mommert 2020 PDF.

## Decisions encoded (from brainstorming dialogue, 2026-04-25)

Each decision below was an explicit choice during the brainstorming pass.
Reproduced here so the implementer can refer back to a single source of
truth without re-reading the conversation.

| # | Decision | Choice |
|---|---|---|
| 1 | Phase 1 decomposition | Three sequential specs aligned to dependencies; this spec covers §1.1 only |
| 2 | Spec 1 deliverable boundary | Code + orchestrator + iteration runbook; orchestrator drives one task end-to-end with a single command on the GPU box |
| 3 | Parity pass/fail thresholds | **Standard tier:** cls acc ±2pp, seg IoU ±0.03, seg img-acc ±2pp, area-ratio one-sided ≤ paper + 0.03; AUC ungated (paper doesn't report) |
| 3b | Iteration runbook on FAIL | One retry with strict-paper hyperparameters; if still FAIL, document and accept as new baseline (append "Known parity gap" section to `PAPER_PARITY_AUDIT.md`); tag `v0.3.0` regardless of PASS/FAIL |
| 4 | "Signed JSON" interpretation | **Provenance-rich, not cryptographic.** Embedded git/config/checkpoint/dataset metadata; signed numerical deltas; no PGP/sigstore (Phase 3.6 territory) |
| 5 | Hyperparameter audit source | **Both sources, per-knob audit table doc** at `docs/PAPER_PARITY_AUDIT.md`. Publication PDF first; HSG-AIML repo `64c806b` second; "?" if neither |
| 6 | Figures callback scope | **Full consolidation.** `figures_callback.py` gains `on_test_end`; `cli/eval.py` loses inline plotting; old eval/ output paths preserved via callback `out_dir_override` |
| 7a | Notebook intent | **CI-runnable from day one.** Papermill parameters + env-var fallback + `SMOKEDET_DEMO=1` mode that uses the synthetic-dataset fixture |
| 7b | Orchestrator | New Makefile targets + `scripts/run_paper_parity.py` Python entry point |
| Architecture | Implementation approach | **Approach 2 (library-first).** New `evaluation/parity.py` and `evaluation/audit.py` are pure-Python (no torch/lightning at import time); `scripts/*` are thin CLI wrappers |
| Audit format | Markdown table; drift risk to YAML accepted for Spec 1 (Phase 2 §2.2 adds CI gate) |  |

## Architecture

### Module dependency graph

```
                   ┌──────────────────────────────────┐
                   │  evaluation/parity.py            │  ← pure library
                   │  - ParityMetric, ParityReport    │     (stdlib + pydantic)
                   │  - evaluate_thresholds()         │     no torch/lightning
                   │  - write_parity_report()         │
                   │  - hash_file(), gather_provenance()
                   └──────────────┬───────────────────┘
                                  │
                ┌─────────────────┼─────────────────┐
                │                 │                 │
        ┌───────▼──────┐  ┌───────▼─────────┐ ┌─────▼─────────────────┐
        │ scripts/     │  │ scripts/         │ │ (future) §1.6        │
        │ report_parity│  │ run_paper_parity │ │ scripts/run_ablation │
        │ .py          │  │ .py              │ │ — imports same lib    │
        │ (CLI wrapper)│  │ (orchestrator)   │ └───────────────────────┘
        └──────┬───────┘  └────────┬─────────┘
               │                   │
               │            ┌──────▼──────────────┐
               │            │ smoke_detection.cli.│
               │            │ train + eval        │
               │            │ (existing)          │
               │            └──────┬──────────────┘
               │                   │
        ┌──────▼───────────────────▼──────┐
        │ training/figures_callback.py    │  ← gains on_test_end
        │ (existing, modified)             │     attached by both train.py and eval.py
        └──────────────────────────────────┘

        ┌──────────────────────────────┐
        │ evaluation/audit.py          │  ← pure library
        │  - load_audit_table()        │     reads docs/PAPER_PARITY_AUDIT.md
        │  - validate_against_yaml()   │     compares vs. configs/*/paper.yaml
        └──────────────┬───────────────┘
                       │ (used by audit unit test; later by Phase 2 CI gate)
```

**Key boundary:** `evaluation/parity.py` does not import torch or lightning.
Callers pass a `dict[str, float]` of metrics in; the library returns a
serializable `ParityReport`. This is what makes it unit-testable on
GPU-less machines and what lets §1.6's ablation runner consume it without
shape contortions.

### File-level diff

**New files:**
```
src/smoke_detection/evaluation/parity.py
src/smoke_detection/evaluation/audit.py
scripts/run_paper_parity.py
docs/PAPER_PARITY_AUDIT.md
docs/RUNBOOK_paper_parity.md
tests/unit/evaluation/test_parity.py
tests/unit/evaluation/test_audit.py
tests/unit/training/test_figures_callback.py
tests/integration/test_run_paper_parity.py
tests/integration/test_notebook_smoke.py
```

**Modified files:**
```
src/smoke_detection/training/figures_callback.py
src/smoke_detection/cli/eval.py
scripts/report_parity.py
configs/classification/paper.yaml
configs/segmentation/paper.yaml
Makefile
notebooks/results.ipynb
```

**Untouched (explicitly):** `data/transforms.py`, datamodules, datasets,
`models/*.py`, LightningModules, `configs/loader.py`, `configs/*.py` (the
schemas), `scripts/prepare_dataset.py`, `tests/conftest.py`.

### Data flow — one orchestrator run

```
GPU box: make parity-cls
  └─> python scripts/run_paper_parity.py --task classification \
          --config configs/classification/paper.yaml
       │
       ├─ Phase A: TRAIN (subprocess: cli/train.py)
       │   produces: lightning_logs/<exp>/version_N/
       │              ├ checkpoints/last.ckpt + best
       │              └ figures/training_curves.png + val_predictions.png
       │
       ├─ Phase B: TEST + PARITY REPORT (in-process, single pass)
       │   build Trainer(logger=False, callbacks=[figures_cb_with_override]),
       │   load checkpoint, run trainer.test() once to:
       │     - get the metrics dict (Trainer.test return value)
       │     - emit eval figures via figures_callback.on_test_end
       │       → lightning_logs/<exp>/version_N/eval/{confusion_matrix,roc_curve}.png
       │   then call evaluation.parity.write_parity_report(...)
       │   writes: lightning_logs/<exp>/version_N/parity.json
       │   prints: stdout table
       │
       └─ Exit: 0 if overall.pass else 2; 1 on unexpected error
          (Orchestrator does NOT auto-retry; runbook drives the iteration loop.)
          Note: cli/eval.py is NOT subprocessed — the orchestrator does the
          equivalent work in-process to avoid running the test set twice.
          cli/eval.py remains as a user-facing CLI for ad-hoc evaluation.
```

## Component design

### §1 — `evaluation/parity.py` (the library)

#### Threshold constants

```python
PAPER: dict[str, dict[str, float | None]] = {
    "classification": {
        "test_accuracy": 0.943,   # Mommert 2020 abstract
        "test_auc":      None,    # not reported in abstract; ungated
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
    "classification": {"test_accuracy": "higher_better", "test_auc": "higher_better"},
    "segmentation":   {"test_iou": "higher_better",
                       "test_img_accuracy": "higher_better",
                       "mean_abs_area_ratio_error": "lower_better"},
}
```

A metric with `paper=None` or `tolerance=None` is **ungated** — its value
is recorded for trend analysis but cannot trigger FAIL.
`mean_abs_area_ratio_error` is one-sided (lower is better): the gate is
`value ≤ paper + tolerance`, not `|value − paper| ≤ tolerance`.

#### Pydantic types

```python
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime

Direction = Literal["higher_better", "lower_better"]

class ParityMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value:     float | None
    paper:     float | None
    delta:     float | None       # value - paper
    tolerance: float | None
    direction: Direction = "higher_better"
    pass_:     bool | None = Field(alias="pass")  # None = ungated

class ParityProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    git_commit:        str
    git_dirty:         bool
    config_path:       str
    config_sha256:     str
    checkpoint_path:   str | None
    checkpoint_sha256: str | None
    dataset_root:      str
    dataset_file_count: dict[str, int]   # {"train": N, "val": N, "test": N}

class ParityOverall(BaseModel):
    model_config = ConfigDict(extra="forbid")
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

#### Public functions

```python
def evaluate_thresholds(
    task: Literal["classification", "segmentation"],
    observed: Mapping[str, float],
) -> tuple[dict[str, ParityMetric], ParityOverall]: ...

def gather_provenance(
    config_path: Path,
    checkpoint_path: Path | None,
    dataset_root: Path,
    dataset_file_count: dict[str, int],
) -> ParityProvenance: ...

def write_parity_report(
    task: Literal["classification", "segmentation"],
    observed: Mapping[str, float],
    config_path: Path,
    checkpoint_path: Path | None,
    dataset_root: Path,
    dataset_file_count: dict[str, int],
    out_path: Path,
) -> ParityReport: ...

def hash_file(path: Path, *, chunk_size: int = 1 << 20) -> str: ...
def git_state() -> tuple[str, bool]: ...   # (commit_sha, is_dirty); never raises
def format_table(report: ParityReport) -> str: ...   # stdout summary

# Translation from Lightning metric keys (slash-prefixed) to canonical
# snake_case names used in PAPER/TOLERANCE/DIRECTION + the JSON schema.
# Used by the orchestrator (and future ablation runner) to bridge
# Trainer.test() return shape into the library's contract.
LIGHTNING_KEY_MAP: dict[str, str] = {
    "test/acc":     "test_accuracy",
    "test/auc":     "test_auc",
    "test/iou":     "test_iou",
    "test/img_acc": "test_img_accuracy",
    # mean_abs_area_ratio_error is computed manually by callers (segmentation only;
    # not exposed by Trainer.test), passed in directly under that canonical name
}

def translate_lightning_metrics(
    raw: Mapping[str, float],
) -> dict[str, float]:
    """Apply LIGHTNING_KEY_MAP; pass through keys that are already canonical;
    drop unknown keys with a debug log."""
```

#### Edge cases

- **NaN/inf observed value:** `value=NaN`, `delta=None`, `pass_=False`,
  contributes to `failed_count`. Hard fail — a NaN metric is almost
  certainly a bug, not a parity gap.
- **Observed metric not in `PAPER[task]`:** logged at debug, omitted
  from report (keeps schema stable across Lightning/torchmetrics drift).
- **`PAPER[task]` metric missing from observed:** written with
  `value=None, delta=None, pass_=None`, contributes to `ungated_count`,
  logged at warning.
- **Git unavailable:** `git_commit="unknown"`, `git_dirty=True`. Doesn't crash.
- **Checkpoint file missing:** `checkpoint_path=None,
  checkpoint_sha256=None`. For ablation runs that report on a metrics
  dict without a saved checkpoint.

#### Example output

```json
{
  "schema_version": 1,
  "task": "classification",
  "produced_at": "2026-04-25T22:14:07+00:00",
  "provenance": {
    "git_commit": "1aeaa75",
    "git_dirty": false,
    "config_path": "configs/classification/paper.yaml",
    "config_sha256": "8c3f...",
    "checkpoint_path": "lightning_logs/.../checkpoints/last.ckpt",
    "checkpoint_sha256": "e2a1...",
    "dataset_root": "/data/zenodo/4250706_prepared",
    "dataset_file_count": { "train": 8421, "val": 1804, "test": 1805 }
  },
  "metrics": {
    "test_accuracy": { "value": 0.927, "paper": 0.943, "delta": -0.016,
                       "tolerance": 0.02, "direction": "higher_better", "pass": true },
    "test_auc":      { "value": 0.981, "paper": null, "delta": null,
                       "tolerance": null, "direction": "higher_better", "pass": null }
  },
  "overall": { "pass": true, "passed_count": 1, "failed_count": 0, "ungated_count": 1 }
}
```

### §2 — `evaluation/audit.py` + `docs/PAPER_PARITY_AUDIT.md`

#### Audit doc shape

A single committed markdown file with two tables (one per task). Every
YAML knob in `configs/{cls,seg}/paper.yaml` gets one row; tolerance rows
for each gated metric also live here so the threshold is documented next
to the hyperparameter that produced it.

```markdown
# Paper Parity Audit

Authoritative comparison of `configs/{classification,segmentation}/paper.yaml`
against (a) Mommert et al. 2020, NeurIPS workshop and (b) the pre-Lightning
HSG-AIML repo at commit `64c806b`.

**Source priority:** publication PDF first; HSG-AIML repo second; "?" if neither.

**Decision rule on disagreement:** prefer the publication; document the choice
in the `notes` column.

## Classification (`configs/classification/paper.yaml`)

| section.key             | yaml_value | paper_value | hsg_aiml_value | source       | notes |
|-------------------------|------------|-------------|----------------|--------------|-------|
| trainer.max_epochs      | 100        | ?           | 100            | hsg-aiml     | verify-against-paper |
| optim.lr                | 0.3        | ?           | 0.3            | hsg-aiml     | verify-against-paper |
| optim.momentum          | 0.7        | ?           | 0.7            | hsg-aiml     | unusual; verify-against-paper |
| optim.weight_decay      | 0.0        | ?           | 0.0            | hsg-aiml     | verify-against-paper |
| optim.scheduler         | plateau    | ?           | plateau        | hsg-aiml     | verify-against-paper |
| model.backbone          | resnet50   | ?           | resnet50       | hsg-aiml     | verify-against-paper |
| model.pretrained        | true       | ?           | true           | hsg-aiml     | verify-against-paper |
| model.in_channels       | 4          | ?           | 4              | hsg-aiml     | verify-against-paper; paper also reports 12-ch |
| trainer.precision       | "32"       | ?           | float32        | hsg-aiml     | not stated; HSG-AIML default |
| data.batch_size         | 30         | ?           | 30             | hsg-aiml     | verify-against-paper |
| data.crop_size          | 90         | ?           | 90             | hsg-aiml     | verify-against-paper |
| data.balance            | upsample   | ?           | upsample       | hsg-aiml     | verify-against-paper |
| **gate**: test_accuracy | tol ±0.02  | 0.943       | —              | spec         | per Spec 1 question 3 (Standard) |
| **gate**: test_auc      | ungated    | —           | —              | spec         | not reported in publication |

## Segmentation (`configs/segmentation/paper.yaml`)

(same shape, with seg-specific knobs and three gates: test_iou,
test_img_accuracy, mean_abs_area_ratio_error)
```

**Initial conservative state:** because the user does not currently have
PDF access, every "could-be-in-paper" row starts as
`paper_value="?", source="hsg-aiml", notes="verify-against-paper"`. The
runbook (§7) includes a post-Spec-1 cleanup task: when PDF access is
restored, walk every `verify-against-paper` row and upgrade
`paper_value` + `source` (and possibly `notes` with a `paper §X.Y`
citation).

#### YAML annotations (lightweight, not parsed)

`paper.yaml` files get one inline comment per knob naming the source:

```yaml
optim:
  lr: 0.3                    # source: hsg-aiml@64c806b (verify-against-paper)
  momentum: 0.7              # source: hsg-aiml@64c806b (verify-against-paper)
  weight_decay: 0.0          # source: hsg-aiml@64c806b (verify-against-paper)
  scheduler: plateau         # source: hsg-aiml@64c806b (verify-against-paper)
```

Pure documentation; the audit parser doesn't read these.

#### `audit.py` API

```python
class AuditRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key:            str                # "optim.lr"
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

def load_audit_table(
    md_path: Path = Path("docs/PAPER_PARITY_AUDIT.md"),
) -> dict[str, AuditTable]: ...

def validate_against_yaml(
    table: AuditTable, yaml_path: Path,
) -> list[str]: ...   # discrepancy strings; empty = clean

def validate_all(
    md_path: Path = Path("docs/PAPER_PARITY_AUDIT.md"),
    cls_yaml: Path = Path("configs/classification/paper.yaml"),
    seg_yaml: Path = Path("configs/segmentation/paper.yaml"),
) -> list[str]: ...
```

#### Validation semantics

`validate_against_yaml` compares **rendered string** values, not typed
values, because the table is markdown. A `_render(value: Any) -> str`
helper canonicalizes: `True/False → "true"/"false"`, `None → "null"`,
strings unquoted, numbers via `repr`. The same renderer is documented in
the audit doc as the canonical rendering rule.

#### Parser

~50 lines of stdlib. Anchor on `## Classification (`/`## Segmentation (`
headers; find the next pipe-table; split on `|`; map columns by header
name (so column reorderings don't break parsing). No third-party markdown
library.

`is_gate=True` is detected by a `**gate**:` prefix in the `section.key`
column (matches the audit doc's rendering convention). The parser
strips the prefix when populating `AuditRow.key` so callers see clean
names like `test_accuracy` instead of `**gate**: test_accuracy`.

#### Spec 1 doesn't add a CI gate

The audit doc + audit unit test catch drift the moment a developer
changes the YAML or the doc and forgets the other. Phase 2 §2.2 promotes
this into a CI build-blocking step; Spec 1 just provides the library
and the test.

### §3 — `figures_callback.py` consolidation

#### Current behavior

`TrainingFiguresCallback.on_train_end` writes `training_curves.png` +
`val_predictions.png` to `<logger.log_dir>/figures/`. Wired in
`cli/train.py`. `cli/eval.py` produces its own `confusion_matrix.png`,
`roc_curve.png`, `iou_distribution.png`, `area_ratio_distribution.png`
inline (duplicating what a callback could do).

#### Target behavior

Callback owns figure generation across both train and test. `cli/eval.py`
becomes thin: build datamodule, build module, build trainer with
callback attached, call `trainer.test()`, return.

#### Changes to `figures_callback.py`

```python
class TrainingFiguresCallback(Callback):
    def __init__(
        self,
        num_val_samples: int = 9,
        out_dir_override: Path | None = None,   # NEW
    ):
        ...
        self._out_dir_override = out_dir_override
        # NEW: per-batch test-time accumulators
        self._test_scores: list[float] = []
        self._test_labels: list[int] = []
        self._test_tp = self._test_tn = self._test_fp = self._test_fn = 0
        self._test_ious: list[float] = []
        self._test_area_ratios: list[float] = []

    def _resolve_out_dir(self, trainer):
        if self._out_dir_override is not None:
            return self._out_dir_override
        # existing logic: <logger.log_dir>/figures or <default_root_dir>/figures
        ...

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # mirror per-batch logic currently in cli/eval.py _eval_classification/_eval_segmentation
        ...

    def on_test_end(self, trainer, pl_module):
        out_dir = self._resolve_out_dir(trainer)
        out_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(pl_module, ClassificationModule):
            plot_confusion_matrix(self._test_tp, self._test_tn, self._test_fp,
                                  self._test_fn, out_dir / "confusion_matrix.png")
            plot_roc_curve(self._test_scores, self._test_labels,
                           out_dir / "roc_curve.png")
        elif isinstance(pl_module, SegmentationModule):
            plot_iou_distribution(self._test_ious,
                                  out_dir / "iou_distribution.png")
            plot_area_ratio_distribution(self._test_area_ratios,
                                         out_dir / "area_ratio_distribution.png")

    # NEW: public accessor for derived test metrics that aren't produced
    # by the LightningModule's test_step / Trainer.test return value.
    # The orchestrator merges this dict with the Trainer.test return
    # dict before passing to evaluation.parity.evaluate_thresholds.
    def extra_test_metrics(self) -> dict[str, float]:
        """Return derived test-time metrics computed by this callback.
        Currently: mean_abs_area_ratio_error (segmentation only).
        Returns {} if no test pass has run or the module type doesn't
        produce additional derived metrics."""
        if not self._test_area_ratios:  # no test pass run, or not segmentation
            return {}
        # mean(|1 - (a_pred / a_true)|) — matches existing report_parity.py
        # area-ratio computation. Skips ratios where a_true was 0
        # (handled in on_test_batch_end accumulation).
        import numpy as np
        return {"mean_abs_area_ratio_error": float(np.mean(
            [abs(1.0 - r) for r in self._test_area_ratios]
        ))}
```

The accumulation logic is **moved verbatim** from `cli/eval.py` — same
math, just relocated. No semantic change. The new
`extra_test_metrics()` method exposes derived metrics that
`Trainer.test()` does not include in its return value, so the
orchestrator can merge them without crossing encapsulation boundaries.

#### Changes to `cli/eval.py`

```python
def _eval_classification(cfg, ckpt, out_dir):
    dm = ClassificationDataModule(...)
    module = ClassificationModule.load_from_checkpoint(str(ckpt), weights_only=False)
    figures_cb = TrainingFiguresCallback(out_dir_override=out_dir)
    trainer = Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[figures_cb],
        logger=False,
    )
    trainer.test(module, datamodule=dm)
    # No more inline plotting; no manual loop over test_dataloader.
```

Removes ~60 lines of duplicate per-batch + plotting code from
`cli/eval.py`. `out_dir` retains its current default
(`<output_dir>/<experiment_name>/eval/`) so external scripts/notebooks
keep working.

### §4 — `notebooks/results.ipynb` (CI-runnable, papermill-friendly)

#### Cell structure

1. **Markdown — Title & description.**
2. **Parameters cell** (tagged `parameters` for papermill):
   ```python
   task = "classification"
   parity_json_path = os.environ.get(
       "SMOKEDET_PARITY_JSON",
       "lightning_logs/classification_4ch_resnet50_paper/version_0/parity.json",
   )
   figures_dir = os.environ.get(
       "SMOKEDET_FIGURES_DIR",
       "lightning_logs/classification_4ch_resnet50_paper/version_0/eval",
   )
   checkpoint_path = os.environ.get(
       "SMOKEDET_CHECKPOINT_PATH",
       "lightning_logs/classification_4ch_resnet50_paper/version_0/checkpoints/last.ckpt",
   )
   demo_mode = os.environ.get("SMOKEDET_DEMO", "0") == "1"
   ```
3. **Setup cell.** Imports; if `demo_mode`, build the synthetic-dataset
   fixture + run `fast_dev_run` train + eval to populate artifacts the
   notebook reads. Otherwise just load existing artifacts.
4. **Markdown — Provenance & parity summary.** Renders `provenance` and
   `overall` blocks of `parity.json` as a markdown table.
5. **Per-metric table cell.** Styled table; PASS/FAIL color-coded.
6. **Figure embed cells.** `IPython.display.Image` against the PNGs in
   `figures_dir` and `<log_dir>/figures/`.
7. **Per-image qualitative samples.** Embed `val_predictions.png` from
   `<log_dir>/figures/` (already produced by the figures callback's
   `on_train_end`). The notebook does not call private functions on the
   callback; it just renders artifacts the callback already wrote.
8. **Markdown — Capstone narrative.** Boilerplate text + placeholder
   `<TBD: parity gap discussion>` paragraphs the user fills in after the
   GPU run.

#### Demo mode

`SMOKEDET_DEMO=1` enables a self-sufficient mode using
`tests._data.build_synthetic_prepared_tree` + `fast_dev_run`. ≈30 seconds
on CPU. If `demo_mode=0` and artifacts are missing, the notebook displays
a clear "no parity.json found at <path> — run `make parity-cls` first"
markdown cell instead of crashing.

### §5 — Orchestrator + runbook

#### `scripts/run_paper_parity.py`

```
Usage:
    python scripts/run_paper_parity.py --task classification \
        --config configs/classification/paper.yaml

Optional:
    --skip-train             use existing checkpoint at
                             <output>/<exp>/version_<N>/checkpoints/last.ckpt
    --version N              force log version
    --strict-paper           refuse to start unless paper.yaml validates clean
                             against PAPER_PARITY_AUDIT.md
    --paper-overrides PATH   test-only; JSON overriding PAPER+TOLERANCE
                             (used by the integration test against synthetic
                             data; never use in real parity runs)
```

**Flow:**

1. Parse args; validate `--task ∈ {classification, segmentation}`.
2. Load config via `configs.loader.load_config`.
3. If `--strict-paper`, run
   `evaluation.audit.validate_against_yaml(table_for_args.task, paper_yaml_for_args.task)`;
   exit 1 with discrepancy list if not clean. (Validates only the
   table for the orchestrator's `--task`; the other task's row drift
   is the other orchestrator invocation's problem.)
4. Resolve / create `lightning_logs/<exp>/version_N/`.
5. Unless `--skip-train`, subprocess `python -m smoke_detection.cli.train
   --config <config>`. Propagate non-zero exits.
6. Locate `checkpoints/last.ckpt` in the resolved version dir; exit 1 if missing.
7. **In-process (single test pass):** build
   ```python
   figures_cb = TrainingFiguresCallback(
       out_dir_override=<version_dir>/"eval"
   )
   trainer = Trainer(logger=False, callbacks=[figures_cb], ...)
   results = trainer.test(module, datamodule=dm)  # list[dict]
   ```
   This single `trainer.test()` call:
     - emits eval figures via `figures_cb.on_test_end`
     - returns the Lightning-keyed metrics dict in `results[0]`
8. Build the canonical metrics dict for the parity library:
   ```python
   raw = results[0]                                     # {"test/acc": ..., "test/auc": ...}
   canonical = translate_lightning_metrics(raw)        # {"test_accuracy": ..., "test_auc": ...}
   canonical.update(figures_cb.extra_test_metrics())   # adds mean_abs_area_ratio_error for seg
   ```
9. Walk dataset for file counts (cheap; sub-second `os.walk` over the
   three split directories).
10. `evaluation.parity.write_parity_report(task, canonical, ...)` writing
    `<version_dir>/parity.json`.
11. Print `format_table(report)`.
12. Exit `0` if `report.overall.pass else 2`.

**Why step 5 subprocesses but step 7 is in-process:** keeps `cli/train.py`
as the canonical user-facing CLI for training (the runbook documents the
copyable invocation; reuses the trainer setup including TensorBoard
logger). Step 7 needs the Python metrics dict back, which is awkward
across a subprocess boundary; the in-process Trainer.test with the
callback attached produces both figures and metrics in a single pass.
`cli/eval.py` is not subprocessed by the orchestrator; it remains a
user-facing CLI for ad-hoc evaluation outside the parity flow.

#### Makefile additions

```makefile
.PHONY: parity-cls parity-seg parity

parity-cls:
	python scripts/run_paper_parity.py --task classification \
	    --config configs/classification/paper.yaml

parity-seg:
	python scripts/run_paper_parity.py --task segmentation \
	    --config configs/segmentation/paper.yaml

# Exit 2 from orchestrator counts as failure here, halting the chain.
parity: parity-cls parity-seg
```

#### `docs/RUNBOOK_paper_parity.md`

Sections:

1. **Pre-flight checklist:** CUDA available; dataset prepared (link to
   `prepare_dataset.py` invocation); clean git status; `uv sync --extra dev`.
2. **Single-command happy path:** `make parity` (≈ 4–10 hrs on a single
   modern GPU).
3. **If `parity-cls` exits 2:** inspect `parity.json`; re-run with
   `--strict-paper`; if still FAIL, append "Known parity gap" section to
   `PAPER_PARITY_AUDIT.md` and accept; per Spec 1 question 3b, do not loop.
4. **Same logic for `parity-seg`** — segmentation typically takes longer
   (300 epochs) — consider running overnight.
5. **Capture artifacts:** commit `parity.json` files; re-run notebook
   via `papermill notebooks/results.ipynb out.ipynb -p task classification
   -p parity_json_path lightning_logs/.../parity.json`; review; commit.
   **Verify-against-paper pass:** when PDF access is restored, walk every
   `verify-against-paper`-noted row in `PAPER_PARITY_AUDIT.md` and
   upgrade with `paper §X.Y` citations.
6. **Tag release:** `git tag v0.3.0 && git push --tags` (ask before pushing).
7. **Failure escape hatches:**
   - OOM at paper batch size → lower batch size, document deviation, accept gap
   - Training diverges (NaN loss) at `lr=0.3` → add
     `--override trainer.gradient_clip_val=1.0`, document deviation

## Tests

| File | Coverage |
|---|---|
| `tests/unit/evaluation/test_parity.py` | `evaluate_thresholds` (PASS, FAIL, ungated, NaN/inf, missing metric, lower-better one-sided gate); `gather_provenance` (file hashing, git state, dirty/clean); `write_parity_report` (JSON round-trip, schema validation); `format_table` (snapshot) |
| `tests/unit/evaluation/test_audit.py` | `load_audit_table` (parses both sections); `validate_against_yaml` (clean + intentionally-broken-YAML cases); `validate_all` (end-to-end against the committed PAPER_PARITY_AUDIT.md) |
| `tests/unit/training/test_figures_callback.py` | `on_train_end` writes both PNGs after `fast_dev_run` fit; `on_test_end` writes correct PNGs per task type; empty-test-loader doesn't crash; `out_dir_override` honored |
| `tests/integration/test_run_paper_parity.py` | Run orchestrator end-to-end with `--config <synthetic-config> --paper-overrides <permissive.json>` against synthetic fixture, `fast_dev_run` for both train and eval. Assert `parity.json` produced with expected schema; orchestrator exits 0 |
| `tests/integration/test_notebook_smoke.py` | `jupyter nbconvert --execute notebooks/results.ipynb` with `SMOKEDET_DEMO=1`; assert notebook runs to completion without errors |

`--paper-overrides` flag exists *only* for the integration test (synthetic
data won't hit 94.3% accuracy). Documented as test-only.

Spec 1 doesn't introduce a coverage gate (Phase 2 §2.3 does). New code
should hit ≥80% locally without further effort.

## Deliverables

### Committed by Spec 1 implementation (no-GPU box)

```
NEW
src/smoke_detection/evaluation/parity.py
src/smoke_detection/evaluation/audit.py
scripts/run_paper_parity.py
docs/PAPER_PARITY_AUDIT.md
docs/RUNBOOK_paper_parity.md
tests/unit/evaluation/test_parity.py
tests/unit/evaluation/test_audit.py
tests/unit/training/test_figures_callback.py
tests/integration/test_run_paper_parity.py
tests/integration/test_notebook_smoke.py

MODIFIED
src/smoke_detection/training/figures_callback.py   (on_test_end + accumulators + out_dir_override)
src/smoke_detection/cli/eval.py                    (thinned; callback-driven plots)
scripts/report_parity.py                           (CLI wrapper over evaluation.parity)
configs/classification/paper.yaml                  (per-knob source comments)
configs/segmentation/paper.yaml                    (per-knob source comments)
Makefile                                           (parity-cls / parity-seg / parity)
notebooks/results.ipynb                            (parameterized, papermill+env-var, demo_mode)
```

### Committed by Spec 1 execution (GPU box)

```
NEW
lightning_logs/classification_4ch_resnet50_paper/version_<N>/parity.json
lightning_logs/segmentation_4ch_unet_paper/version_<N>/parity.json
docs/PAPER_PARITY_AUDIT.md  (potentially appended with "Known parity gap" section)
notebooks/results.ipynb     (re-committed with populated narrative/numbers)
```

The two `.ckpt` files stay on the GPU box and are not committed (size +
existing `.gitignore` policy). Their `checkpoint_sha256` lives in
`parity.json` so reproducibility is recoverable.

### Release

- Tag `v0.3.0` on the commit that includes both `parity.json` files
- CHANGELOG entry: paper-parity baseline established; per-task PASS/FAIL/gap
  status; pointers to `RUNBOOK_paper_parity.md` and `PAPER_PARITY_AUDIT.md`
- Tagging is a runbook step, not a CI step (Phase 2 §2.2 adds CI gates)

## Success criteria

This spec is complete when, on `phase-1-generalization`:

1. All code-only deliverables land and `pytest` passes on the no-GPU box
2. The user executes the runbook on the GPU box and produces two
   `parity.json` artifacts (one per task), each either marked overall
   PASS or with a documented "Known parity gap" section appended to
   `PAPER_PARITY_AUDIT.md`
3. Both checkpoints exist on the GPU box and their hashes are recorded
   in the respective `parity.json` files
4. `notebooks/results.ipynb` has been re-run and committed with the GPU
   results
5. `v0.3.0` tag exists on the commit containing both `parity.json` files

## Out of scope (explicit)

- Anything in §1.2–§1.6 (Spec 2 and Spec 3)
- Cryptographic signatures (PGP/sigstore) on `parity.json` — Phase 3.6
- CI build-blocking parity gate — Phase 2 §2.2
- `nbstripout` pre-commit + notebook-CI workflow — Phase 2 §2.4 (Spec 1
  *enables* CI-runnability via parameterization but doesn't *enforce* it)
- Changes to model code, loss, optimizer choice, dataset preparation,
  or the augmentation pipeline (those are Spec 3)

## Open questions / risks

- **PDF access for verification pass.** Audit doc starts conservative
  (`source: hsg-aiml`, `note: verify-against-paper`). Real PDF citations
  arrive in a follow-up cleanup pass (runbook §5). Risk: if verification
  reveals a hyperparameter mismatch, we may need to re-run parity. Likely
  small.
- **Paper parity may not be achievable.** Per parent design rationale's
  open questions: "If parity gap is >2% accuracy, decide: accept as new
  baseline, or revisit the 4-channel design." Spec 1's iteration runbook
  picks "accept and document" after one retry; Spec 2 may revisit this if
  generalization improvements depend on the design choice.
- **Lightning version drift.** `Trainer.test()` return shape is what
  `evaluate_thresholds` consumes via key lookup. If torchmetrics or
  Lightning rename `test/acc` to `test_acc` (or similar), the warning
  log fires and metrics quietly become ungated. Phase 2's CI gate would
  catch this; Spec 1 relies on the unit test running against pinned
  Lightning + the integration test exercising the actual key shape.
- **Subprocess boundary in orchestrator.** `subprocess.run` on
  `cli/train.py` means the orchestrator's stderr/stdout interleaving
  isn't perfect. Acceptable: training output goes to TensorBoard +
  `lightning_logs/`; orchestrator print is just summary lines.

## References

- Parent roadmap design: [`2026-04-25-todo-roadmap-design.md`](2026-04-25-todo-roadmap-design.md)
- `TODO.md` Phase 1 §1.1
- Mommert et al. 2020, *Characterization of Industrial Smoke Plumes
  from Remote Sensing Data*, NeurIPS Tackling Climate Change with ML
  workshop (Zenodo dataset DOI: 10.5281/zenodo.4250706)
- Existing `scripts/report_parity.py` (will be thinned)
- Existing `src/smoke_detection/training/figures_callback.py`
  (will gain `on_test_end`)
- `docs/augmentation-improvements.md` — referenced by §1.3 audit
  (Spec 3 territory; Spec 1 does not touch this)
