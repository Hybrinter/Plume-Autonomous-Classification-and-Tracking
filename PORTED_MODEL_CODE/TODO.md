# TODO

Phased roadmap for the Industrial Smoke Plume Detection repo. This repo
develops and validates the ResNet-50 classifier and U-Net segmenter; the
deployed model targets a **Nvidia Jetson Orin AGX** running Rust + `ort`
/ TensorRT in the flight-software stack.

**Priority order (phases run sequentially):**

1. **Generalization** — survive OOD on-orbit data
2. **Engineering Health** — guard generalization improvements from regressions
3. **Production / Handoff** — TensorRT-ready artifacts for the Rust consumer
4. **On-Orbit Continuous Learning** — closed-loop downlink → label → retrain → re-export

**Inference budget constraint:** The Orin AGX is shared with other satellite
workloads. No multi-forward-pass inference techniques (TTA, MC-dropout,
deep ensembles) in the deployed path — eval-time diagnostics only.

See `docs/superpowers/specs/2026-04-25-todo-roadmap-design.md` for design rationale.

---

## Phase 1 — Generalization (current focus)

### 1.1 Establish the baseline (gating)

- [ ] Train classifier (ResNet-50) on Zenodo to paper parity
- [ ] Train segmenter (U-Net) on Zenodo to paper parity
- [ ] Extend `scripts/report_parity.py` to emit a signed JSON report
      (acc/AUC/IoU, deltas vs. Mommert 2020 Table 1, pass/fail thresholds)
- [ ] Verify `configs/{classification,segmentation}/paper.yaml` hyperparameters
      match the publication
- [ ] Wire `training/figures_callback.py` end-of-eval artifacts +
      regression test that expected files are written
- [ ] Populate `notebooks/results.ipynb` (confusion matrix, ROC, IoU +
      area-ratio histograms, per-image qualitative samples)

### 1.2 Define what generalization means (gating)

- [ ] Build geographic + seasonal + plume-type holdout splits from Zenodo
      metadata; write to `data/splits/`; document in `docs/EVALUATION_PROTOCOL.md`
- [ ] Extend `cli/eval.py` to report metrics per stratum (region, season,
      plume type, cloud-cover bin, mean-radiance bin); emit multi-axis CSV + plots
- [ ] Shortcut-feature audit: train classifier on heavily-blurred inputs
      (σ=8); record finding in `docs/GENERALIZATION_REPORT.md`

### 1.3 Augmentation upgrades (expected-ROI order)

- [ ] Audit existing augmentation wiring — `docs/augmentation-improvements.md`
      references paths that no longer exist post-cleanup; verify
      `SpectralJitter` / `GaussianNoise` are wired in `data/transforms.py`
      and the datamodule; fix or update the doc
- [ ] Add **mixup** (classifier)
- [ ] Add **random erasing / CutOut** (both tasks; segmenter must mask the label region)
- [ ] Add **stronger atmospheric simulation** — low-frequency additive haze
      (Perlin gradient), per-quadrant brightness skew, widen `SpectralJitter` to 0.7–1.3
- [ ] Add **mosaic augmentation** (classifier-only) *[optional, drop if scope is tight]*
- [ ] Add augmentation snapshot tests — fixed-seed pass through transforms,
      pixel-check vs. reference

### 1.4 Inference-time generalization (Orin-budget-aware)

- [ ] **Temperature scaling** — fit on validation split, save alongside checkpoint
- [ ] **OOD / abstention signal** — max-softmax (classifier), mean per-pixel
      max-prob (segmenter); document recommended `--reject-below` threshold
- [ ] **TTA — eval-only** `--tta` flag in `cli/eval.py` (4 rotations × 2 flips
      with proper inverse-transforms); **not** in deployed path

> **Note — MC-dropout is intentionally excluded from the deployed path.**
> Multiplies forward-pass count by N (typically 10–30); unacceptable on
> shared Orin compute. Single-pass max-softmax + entropy is the deployed
> uncertainty signal. (Eval-time MC-dropout for diagnostic comparison may
> be added later as part of §1.6 if useful for the capstone report.)

### 1.5 Training tweaks

- [ ] Stochastic Weight Averaging (Lightning built-in callback, opt-in via config)
- [ ] Label smoothing on classifier (ε=0.05–0.1)

### 1.6 Generalization measurement infrastructure

- [ ] `scripts/run_ablation.py` — drive multi-run config-knob ablation,
      produce comparison report (knob, ID metric, OOD metric, delta)
- [ ] `configs/{classification,segmentation}/paper-robust.yaml` —
      paper baseline + all gen knobs on, **single forward pass on Orin**
- [ ] `configs/classification/paper-robust-eval.yaml` — same weights,
      eval-time techniques on (TTA, etc.) for capstone-report upper-bound only
- [ ] `docs/GENERALIZATION_REPORT.md` — living doc, one row per knob,
      before/after on ID and OOD; capstone evidence

---

## Phase 2 — Engineering Health Re-check

> Gate that makes Phase 1's improvements durable before they ship to Phase 3.

### 2.1 Reproducibility & determinism

- [ ] `torch.use_deterministic_algorithms(True)` audit + document accepted
      non-deterministic ops; set `CUBLAS_WORKSPACE_CONFIG`
- [ ] Reproducibility regression test — train smoketest twice with same seed,
      assert metrics identical to 1e-6
- [ ] Verify `seed_everything` is called early enough in `cli/train.py`

### 2.2 Regression guards

- [ ] Promote `scripts/report_parity.py` into a CI-runnable test with
      explicit pass/fail thresholds
- [ ] Config schema regression — every YAML under `configs/` must round-trip
      through pydantic
- [ ] Memory budget + training-speed assertions on the smoketest config

### 2.3 Coverage & static analysis

- [ ] CI coverage gate ≥80% on `data/transforms.py`, `evaluation/`, `training/`
- [ ] Tests for new Phase-1 code — TTA inverse-transform, temperature scaling,
      OOD threshold computation, mixup mass-conservation, random-erasing
      label coupling, ablation driver
- [ ] **mypy** `--strict` on `data/`, `models/`, `training/`
- [ ] Custom pre-commit hook — scan `docs/*.md` for `src/...` paths and
      validate they exist (catches stale-doc drift automatically)

### 2.4 Notebook & dependency discipline

- [ ] `nbstripout` pre-commit + CI step running `notebooks/results.ipynb`
      end-to-end on a tiny fixture
- [ ] Docker / devcontainer — pinned CUDA, PyTorch, uv lock
- [ ] `uv lock --upgrade --dry-run` audit step in CI
- [ ] "Freeze policy" for `uv.lock` during paper-parity work

### 2.5 Doc health

- [ ] Flesh out `docs/MODEL_ARCHITECTURE.md`
- [ ] Reconcile `docs/augmentation-improvements.md` with current layout
      (or fold into `docs/AUGMENTATION.md`)

---

## Phase 3 — Production / Handoff Readiness (Orin AGX + Rust)

> Produces TensorRT-ready artifacts for the flight-software repo.

### 3.1 Export pipeline

- [ ] `scripts/export_models.py` — both classifier and segmenter
- [ ] ONNX↔PyTorch parity test, `atol=1e-5`
- [ ] `scripts/build_trt_engines.py` — ONNX → TRT for FP32 / FP16 / INT8;
      INT8 needs a 200-image stratified calibration set from validation
- [ ] TRT-vs-PyTorch parity test per precision tier — assert OOD holdout
      accuracy delta ≤ threshold for FP16 and INT8
- [ ] Latency benchmark on Orin (or NVIDIA sim) — p50/p99 at FP16 and INT8;
      include "thermal-throttle to 15 W" datapoint

### 3.2 Model contract & metadata

- [ ] Embed in ONNX: training-config hash, dataset commit, normalization
      stats, channel order, calibrated temperature, recommended classification
      + OOD thresholds
- [ ] Sidecar `model.json` manifest with the same metadata in human-readable form
- [ ] Versioning scheme: `<sha256-weights>` + `<git-sha-config>` +
      `<dataset-commit>` triple — what flight-software pins against

### 3.3 Inference contract documentation

- [ ] Write `docs/INFERENCE_SPEC.md` — channel order, dtype, shape, output
      semantics, threshold guidance (TPR@0.95, FPR@0.01, OOD-reject),
      failure-mode policy (NaN, all-zero, all-saturated)

### 3.4 Failure-mode validation

- [ ] "Rejection input set" — cosmic-ray-hit, lens-flare, all-cloud,
      pure-noise; assert low confidence + high OOD score + no NaN/inf
- [ ] "Input contract" tests — wrong dtype, wrong shape, NaN, all-zero;
      document and assert behavior
- [ ] Thermal-throttle smoke test on Orin — 15 W power mode, latency budget compliance

### 3.5 Rust handoff artifacts

- [ ] Release contents: `classifier.{onnx,engine.fp16,engine.int8}`,
      `segmenter.{onnx,engine.fp16,engine.int8}`, `model.json` per model,
      `expected_io.npy` test vectors
- [ ] Rust-side smoke fixture: tiny `.onnx` + `.npy` input + `.npy`
      expected-output for the flight-software CI
- [ ] `.github/workflows/release.yml` — triggers on `v*` tags, runs
      export + TRT build + parity tests + signs + uploads

### 3.6 Provenance

- [ ] SHA256 + signed git tag per release
- [ ] SBOM (CycloneDX) for training-time dependency closure
- [ ] Document training-data provenance: Zenodo DOI, file list, seed sequence,
      training run date

---

## Phase 4 — On-Orbit Continuous Learning (long-term)

> Closed loop: downlink → label → retrain → re-export → uplink.
> Speculative until on-orbit hardware exists; capturing the design now keeps
> Phase 2/3 choices compatible with the eventual loop.

### 4.1 Ingestion

- [ ] Pipeline spec: downlink → ground-station → labeling queue → training-data store
- [ ] Schema: 4-channel TIFF + telemetry sidecar (timestamp, geolocation,
      sensor settings, model output, OOD score)
- [ ] Active-learning prioritizer: rank downlinked images by deployed-model
      OOD score / near-threshold confidence

### 4.2 Labeling

- [ ] Minimal triage UI (Streamlit or Jupyter widget) for the capstone team
- [ ] Label format compatible with classification (binary) and segmentation (mask polygon)
- [ ] Inter-annotator agreement protocol on a subset

### 4.3 Continual training

- [ ] Fine-tuning workflow — load deployed checkpoint, retrain on
      (Zenodo + downlinked) with replay buffer (avoid catastrophic forgetting)
- [ ] Backwards-compat gate — candidate model must not regress on Zenodo OOD
      holdout > threshold before replacing flight model
- [ ] Retraining trigger criteria — e.g., 1000 new labeled images, or drift score > Y

### 4.4 Drift detection (in flight)

- [ ] Telemetry-derived monitoring: distributions of OOD score, confidence,
      input radiance statistics over time
- [ ] Drift threshold definition + operator response playbook

### 4.5 Model registry & deployment

- [ ] Versioned ONNX/TRT registry with provenance, data manifest, evaluation digest
- [ ] Uplink protocol with flight-software team — signed artifact, hash
      verification, "Last Known Good" rollback in flight memory
- [ ] Candidate-model evaluation harness — frozen test suite + backwards-compat
      + operator sign-off
- [ ] "What-if" sandbox — simulate uplink, verify Rust consumer satisfies input contract

### 4.6 Operations & lifecycle

- [ ] Model retirement / sunset policy
- [ ] On-orbit incident playbook — rollback triggers, authorization

---

## Release line mapping

| Phase | Tag | Definition |
|-------|-----|------------|
| 1.1   | v0.3.0 | Paper-parity baseline established |
| 1.2–1.6 | v0.4.0 | Generalization improvements + OOD evaluation harness |
| 2     | v0.5.0 | Engineering health gates in place; CI coverage + mypy + Docker |
| 3     | v1.0.0 | Stable Rust handoff API; ONNX + TRT artifacts under release workflow |
| 4     | v2.x.x | On-orbit continuous learning pipeline |
