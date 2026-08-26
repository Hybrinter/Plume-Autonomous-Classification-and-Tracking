# TODO Roadmap — Phased Restructure

**Status:** approved 2026-04-25
**Supersedes:** the v0.3 / v0.4 / v0.5 milestone framing in `TODO.md` (commit ef8e024)

## Context

The repo just shipped v0.2.0 (Lightning + pydantic + CLI cleanup). The previous
`TODO.md` framed the next year of work as three semver-aligned milestones —
v0.3 (train + paper parity), v0.4 (ONNX export), v0.5 (docs + mypy + Docker).

That framing is correct on the *happy path* but obscures the actual priority
order. The deployed model targets a Nvidia Jetson Orin AGX in a Rust
flight-software stack on a satellite payload, and on-orbit data will be
out-of-distribution relative to the Zenodo training set (different
geographies, atmospheres, scenes — same Sentinel-2-like 4-band
visible+NIR sensor envelope). Generalization work needs to land *before*
ONNX export, not after.

This spec restructures `TODO.md` around four phases that match the actual
priority order:

1. **Generalization** — make the model survive OOD on-orbit data
2. **Engineering Health** — guard generalization improvements from regressions
3. **Production / Handoff** — produce TensorRT-ready artifacts for the
   Rust flight-software consumer
4. **On-Orbit Continuous Learning** — closed-loop downlink → label →
   retrain → re-export

## Constraints

- **Inference target: Nvidia Jetson Orin AGX**, Rust + `ort`/TensorRT,
  shared with other satellite workloads. No multi-forward-pass inference
  techniques (TTA, MC-dropout, deep ensembles) in the deployed path —
  those are eval-time diagnostics only.
- **On-orbit sensor:** same 4-band visible+NIR envelope as Sentinel-2.
  Generalization work targets within-spectral distribution shift, not
  sensor translation.
- **No on-orbit imagery available yet.** Phase 1 assumes Zenodo + synthetic
  augmentation only; pseudo-labeling and domain-adaptation on real on-orbit
  data are deferred to Phase 4.
- **Capstone deadline implicit.** Phase 1 + parts of Phase 2 must be
  complete in time for the capstone report; Phase 3 is the flight-software
  handoff; Phase 4 is post-capstone / on-orbit operations.

## Decision: Approach 2 (restructure)

Three organization options were considered:

1. **Append new sections to existing TODO.md.** Preserves history but
   leaves the existing v0.3 → v0.4 → v0.5 ordering visible, which is
   wrong (export should happen *after* generalization, not as a parallel
   v0.4 milestone). Rejected.

2. **Restructure TODO.md around the four phases.** Replaces the contents
   with phase headings; existing items (paper-parity training, ONNX
   export, mypy, Docker) survive but relocate to the appropriate phase.
   A "Release line" appendix maps phases → tags so the semver story
   isn't lost. **Selected.**

3. **Two-file split (TODO.md + ROADMAP.md).** Two sources of truth
   creates drift hazard for a small project. Rejected.

## Phase 1 — Generalization (current focus)

### Rationale

Paper-parity training is the *baseline*, not the goal. Without it,
no generalization improvement is measurable. After parity, the
bottleneck is twofold: (a) no defined OOD evaluation, so we can't
tell when a change actually generalizes, and (b) the Zenodo dataset
is geographically and seasonally biased — the model may be using
background-context shortcut features (cooling towers, industrial
structure) rather than plume features. Phase 1 fixes both, then
layers in augmentation, inference, and training tweaks in
expected-ROI order.

### 1.1 Establish the baseline (gating)

Carries forward the existing v0.3 items, sharpened.

- Train classifier (ResNet-50) and segmenter (U-Net) on Zenodo to paper parity
- Extend `scripts/report_parity.py` to emit a single signed JSON report
  (acc/AUC/IoU, deltas vs. Mommert Table 1, pass/fail thresholds) — the
  artifact every later improvement is judged against
- Verify `configs/{classification,segmentation}/paper.yaml` hyperparameters
  match the publication
- Wire `training/figures_callback.py` end-of-eval artifacts and add a
  regression test that confirms expected files are written
- Populate `notebooks/results.ipynb` (confusion matrix, ROC, IoU +
  area-ratio histograms, per-image qualitative samples)

### 1.2 Define what generalization means (gating)

No improvement is measurable without these.

- Build geographic + seasonal + plume-type holdout splits from Zenodo
  metadata, write to `data/splits/`, document in
  `docs/EVALUATION_PROTOCOL.md`
- Extend `cli/eval.py` to report metrics per stratum (region, season,
  plume type, cloud-cover bin, mean-radiance bin); emit multi-axis CSV + plots
- **Shortcut-feature audit:** train classifier on heavily-blurred inputs
  (σ=8). If accuracy is anywhere near baseline, the model is leaning on
  low-frequency background — that's the generalization risk worth budgeting
  against. Capture in `docs/GENERALIZATION_REPORT.md`.

### 1.3 Augmentation upgrades (expected-ROI order)

- **Audit first:** `docs/augmentation-improvements.md` references
  `src/smoke_detection/classification/data.py` which no longer exists
  post-cleanup. Verify `SpectralJitter` / `GaussianNoise` are actually
  wired in `data/transforms.py` + the datamodule; fix or update the doc.
- Add **mixup** (classifier) — well-evidenced, low cost
- Add **random erasing / CutOut** (both tasks; for segmenter, also mask
  the label region) — simulates clouds, sensor occlusion
- Add **stronger atmospheric simulation:** low-frequency additive haze
  (Perlin gradient), per-quadrant brightness skew, widen `SpectralJitter`
  to 0.7–1.3
- Add **mosaic augmentation** (classifier-only) — *optional*, drop if
  scope is tight
- Add **augmentation snapshot tests** — serialize a fixed-seed pass through
  transforms, pixel-check vs. reference; catches the same kind of
  pipeline-order bug `augmentation-improvements.md` was written to fix

### 1.4 Inference-time generalization (Orin-budget-aware)

- **Temperature scaling** — single scalar division, essentially free.
  Save temperature alongside checkpoint. The Rust consumer needs
  calibrated probabilities downstream.
- **OOD / abstention signal** — max-softmax (classifier) and mean
  per-pixel max-prob (segmenter); free side-effect of normal inference.
  Document a recommended `--reject-below` threshold for the Rust consumer.
- **TTA — eval-time only.** `--tta` flag in `cli/eval.py` for measuring
  the model's headroom under augmentation. **Not** in the deployed
  inference path (multiplies forward-pass count by 8).
- **MC-dropout — cut from the deployed path.** Multiplies forward-pass
  count by N; unacceptable on shared real-time compute. Replaced by
  temperature-scaled max-softmax + entropy as the single uncertainty signal.

### 1.5 Training tweaks (small-effort, additive)

- Stochastic Weight Averaging — Lightning has a built-in callback,
  add as opt-in via config
- Label smoothing on classifier (ε=0.05–0.1)

### 1.6 Generalization measurement infrastructure

- `scripts/run_ablation.py` — drive N training runs varying one config knob
  at a time, comparison report (knob, ID metric, OOD metric, delta).
  Without this, gen-improvement work is anecdotal.
- `configs/classification/paper-robust.yaml` and
  `configs/segmentation/paper-robust.yaml` — paper baseline + all
  generalization knobs on. **Single-forward-pass on Orin**; no
  inference-time techniques baked in. This is the candidate model for export.
- `configs/classification/paper-robust-eval.yaml` — same weights, all
  eval-time techniques on (TTA, etc.) for measuring upper-bound accuracy
  in the capstone report only.
- `docs/GENERALIZATION_REPORT.md` — living doc, one row per knob,
  before/after on ID and OOD. Becomes capstone evidence.

## Phase 2 — Engineering Health Re-check

### Rationale

Phase 1 touches a lot of code (augmentation, eval, calibration, configs,
ablation infra). Without health checks, regressions slip through into
Phase 3 export silently. Phase 2 is the gate that makes Phase 1's
improvements *durable*.

### 2.1 Reproducibility & determinism

- `torch.use_deterministic_algorithms(True)` audit; document accepted
  non-deterministic ops; set `CUBLAS_WORKSPACE_CONFIG`
- Reproducibility regression test — train smoketest twice with same
  seed, assert metrics identical to 1e-6
- Verify `seed_everything` is called early enough in `cli/train.py`

### 2.2 Regression guards

- Promote `scripts/report_parity.py` into a CI-runnable test with explicit
  pass/fail thresholds
- Config schema regression — every YAML under `configs/` round-trips through pydantic
- Memory budget + training-speed assertions on smoketest

### 2.3 Coverage & static analysis

- CI coverage gate ≥80% on `data/transforms.py`, `evaluation/`, `training/`
- Tests for new Phase-1 code: TTA inverse-transform, temperature-scaling
  fit/apply, OOD threshold computation, mixup mass-conservation,
  random-erasing label coupling, ablation-driver
- **mypy** (was v0.5 — promote here, before more code crystallizes);
  `--strict` on `data/`, `models/`, `training/`
- Custom pre-commit hook: scan `docs/*.md` for `src/...` paths and validate
  they exist (catches `augmentation-improvements.md`-style drift)

### 2.4 Notebook & dependency discipline

- `nbstripout` pre-commit + CI step running `notebooks/results.ipynb`
  end-to-end on a tiny fixture
- Docker / devcontainer (was v0.5): pinned CUDA, PyTorch, uv lock
- `uv lock --upgrade --dry-run` audit in CI
- "Freeze policy" for `uv.lock` during paper-parity work

### 2.5 Doc health

- Flesh out `docs/MODEL_ARCHITECTURE.md` (was v0.5)
- Reconcile `docs/augmentation-improvements.md` with current layout
  (or fold into `docs/AUGMENTATION.md`)

## Phase 3 — Production / Handoff Readiness (Orin AGX + Rust)

### Rationale

Existing v0.4 covered the basics (ONNX, parity, latency, release workflow)
but treated the inference target as generic. With Orin AGX confirmed,
the export path is canonical: ONNX → TensorRT engine, FP16 default,
INT8 with calibration as an option.

### 3.1 Export pipeline

- `scripts/export_models.py` — both models (existing)
- ONNX↔PyTorch parity test, `atol=1e-5` (existing)
- `scripts/build_trt_engines.py` — ONNX → TRT for FP32 / FP16 / INT8;
  INT8 needs a 200-image stratified calibration set from the validation split
- TRT-vs-PyTorch parity per precision tier — assert OOD holdout accuracy
  delta ≤ threshold for FP16 and INT8
- Latency benchmark on Orin (or NVIDIA sim) — p50/p99 at FP16 and INT8,
  plus a "thermal-throttle to 15 W" datapoint

### 3.2 Model contract & metadata

- Embed in ONNX: training-config hash, dataset commit, normalization
  stats, channel order, calibrated temperature, recommended classification
  + OOD thresholds
- Sidecar `model.json` manifest with the same metadata in human-readable form
- Versioning scheme: `<sha256-weights>` + `<git-sha-config>` +
  `<dataset-commit>` triple — what flight-software pins against

### 3.3 Inference contract documentation

- Write `docs/INFERENCE_SPEC.md` (existing, sharpened): channel order,
  dtype, shape, output semantics, threshold guidance (TPR@0.95,
  FPR@0.01, OOD-reject), failure-mode policy (NaN, all-zero, all-saturated)

### 3.4 Failure-mode validation

The single most-overlooked thing in ML→production handoffs.

- "Rejection input set" — cosmic-ray-hit (single bright pixel),
  lens-flare (off-center blob), all-cloud (uniform high brightness),
  pure-noise. Assert low confidence + high OOD score + no NaN/inf.
- "Input contract" tests — wrong dtype, wrong shape, NaN, all-zero —
  document and assert behavior
- Thermal-throttle smoke test on Orin: 15 W power mode latency budget compliance

### 3.5 Rust handoff artifacts

- Release contents: `classifier.{onnx,engine.fp16,engine.int8}`,
  `segmenter.{onnx,engine.fp16,engine.int8}`, `model.json` per model,
  `expected_io.npy` test vectors
- Rust-side smoke fixture: tiny `.onnx` + `.npy` input + `.npy`
  expected-output for the flight-software CI
- `.github/workflows/release.yml` (existing): triggers on `v*` tags,
  runs export + TRT build + parity tests + signs + uploads

### 3.6 Provenance

- SHA256 + signed git tag per release
- SBOM (CycloneDX) for training-time dependency closure
- Document training-data provenance: Zenodo DOI, file list, seed
  sequence, training run date

## Phase 4 — On-Orbit Continuous Learning (long-term)

### Rationale

Closed loop = downlink → label → retrain → re-export → uplink. Mostly
speculative until on-orbit hardware exists, but capturing it now keeps
Phase 2/3 design choices compatible with the eventual loop (manifest
format, versioning triple, OOD-score telemetry).

### 4.1 Ingestion

- Pipeline spec: downlink → ground-station → labeling queue →
  training-data store
- Schema: 4-channel TIFF + telemetry sidecar (timestamp, geolocation,
  sensor settings, model output, OOD score)
- Active-learning prioritizer: rank downlinked images by deployed-model
  OOD score / near-threshold confidence

### 4.2 Labeling

- Minimal triage UI (Streamlit or Jupyter widget) for the capstone team
- Label format compatible with classification (binary) and segmentation
  (mask polygon)
- Inter-annotator agreement protocol on a subset

### 4.3 Continual training

- Fine-tuning workflow: load deployed checkpoint, retrain on
  (Zenodo + downlinked) with replay buffer to avoid catastrophic forgetting
- Backwards-compat gate: candidate model must not regress on Zenodo OOD
  holdout > threshold before replacing flight model
- Retraining trigger criteria: e.g., 1000 new labeled images, or drift > Y

### 4.4 Drift detection (in flight)

- Telemetry-derived monitoring: distributions of OOD score, confidence,
  input radiance statistics over time
- Drift threshold definition + operator response playbook

### 4.5 Model registry & deployment

- Versioned ONNX/TRT registry with provenance, data manifest, evaluation digest
- Uplink protocol with flight-software team: signed artifact, hash
  verification, "Last Known Good" rollback in flight memory
- Candidate-model evaluation harness: frozen test suite + backwards-compat
  + operator sign-off
- "What-if" sandbox: simulate uplink, verify Rust consumer satisfies
  input contract

### 4.6 Operations & lifecycle

- Model retirement / sunset policy
- On-orbit incident playbook (rollback triggers, authorization)

## Release line mapping

The four-phase structure maps to semver tags as follows:

| Phase | Tag    | Definition |
|-------|--------|------------|
| 1.1   | v0.3.0 | Paper-parity baseline established |
| 1.2–1.6 | v0.4.0 | Generalization improvements + OOD evaluation harness |
| 2     | v0.5.0 | Engineering health gates in place; CI coverage + mypy + Docker |
| 3     | v1.0.0 | Stable Rust handoff API; ONNX + TRT artifacts produced under release workflow |
| 4     | v2.x.x | On-orbit continuous learning pipeline |

## Open questions / risks

- **Paper parity may not be achievable in reasonable wall time.** The
  publication uses 12-channel L1C imagery; we use 4 channels post-cleanup.
  If parity gap is >2% accuracy, decide: accept the gap as the new
  baseline, or revisit the 4-channel design choice.
- **Zenodo metadata granularity.** Geographic + seasonal + plume-type
  splits in §1.2 depend on metadata being recoverable from filenames /
  EXIF. If not, this work expands to an Earth-Engine lookup.
- **TensorRT version pinning.** Orin's TRT version drives ONNX opset
  compatibility. Phase 3 export script must target the specific TRT
  version on the flight Orin, not the latest.
- **Capstone team size and labeling capacity.** Phase 4.2 (labeling UI)
  is only worth building if the team has bandwidth to label downlinked
  images. May reduce to "tag + queue, label externally."
- **"Last Known Good" rollback semantics.** Requires coordination with
  the flight-software repo — a write-up needs to align with their boot
  sequence and memory layout.

## References

- Mommert et al. 2020, *Characterization of Industrial Smoke Plumes from
  Remote Sensing Data*, NeurIPS Tackling Climate Change with ML workshop
  (Zenodo dataset DOI: 10.5281/zenodo.4250706)
- Existing TODO.md (commit ef8e024)
- Existing CHANGELOG.md (v0.2.0 release notes)
- `docs/augmentation-improvements.md` (paths need reconciliation per §1.3)
- `docs/MODEL_ARCHITECTURE.md` (stub, fleshed out per §2.5)
