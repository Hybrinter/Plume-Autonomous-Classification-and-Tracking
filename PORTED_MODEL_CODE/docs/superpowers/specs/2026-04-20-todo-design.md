# TODO Roadmap — Design Spec

**Date:** 2026-04-20
**Scope:** Milestone-grouped TODO.md for the Industrial Smoke Plume Detection repo.
Context: capstone project at Colorado School of Mines. This repo develops and
validates the ResNet-50 classifier and U-Net segmenter; trained models are
exported as ONNX files, consumed by a Python flight-software package, and
ultimately repackaged in Rust for HIL testing.

---

## 1. Goals

- Provide a clear, prioritized roadmap from current state (tested, CI-passing
  Lightning pipeline, no trained weights) to flight-software-ready ONNX exports.
- Surface the dependency ordering: `v0.3` (trained weights) must complete before
  `v0.4` (export) can begin; `v0.4` must complete before the flight software
  repo can receive artifacts.
- Include capstone deliverable items (training figures, architecture doc, results
  notebook) without letting them block the critical export path.

---

## 2. Structure

Three milestones, each a `##` section in TODO.md with a checkbox list.

### v0.3.0 — Train & Validate

Gates all downstream work. No export is meaningful without verified weights.

| Item | Rationale |
|---|---|
| Train classifier on Zenodo dataset | No trained weights exist |
| Train segmenter on Zenodo dataset | No trained weights exist |
| Verify paper-accuracy parity | CHANGELOG 0.2.0 explicitly flags this as unverified post-Lightning port |
| Add paper-replication experiment configs | Only `default.yaml` exists; need a config matching the paper's hyperparameters |
| Port training figure generation to Lightning | Apr-16 spec targeted old code; Lightning equivalent is a callback + end-of-eval artifacts |
| Add results notebook | Visualize predictions, confusion matrix, IoU distributions on real validation data |

### v0.4.0 — Export & Integration Interface

Produces the versioned ONNX artifacts consumed by the flight software repo.

| Item | Rationale |
|---|---|
| Add ONNX export script for both models | `ort` is the dominant Rust inference crate; ONNX is the correct format for the Rust repackaging step |
| Add export validation test | Assert ONNX output matches PyTorch output within tolerance (`atol=1e-5`); catches silent precision loss at the graph boundary |
| Write input/output spec doc | Rust consumer needs the exact preprocessing contract: channel order, normalization stats, input shape, output thresholding logic |
| Add batch-1 inference latency benchmark | Real-time gimbal control is latency-sensitive; establish a CPU baseline before the Rust port |
| Add GitHub Releases workflow | CI job triggered on `v*` tags: exports both models, uploads `.onnx` files as release artifacts alongside the wheel |

### v0.5.0 — Documentation & Polish

Capstone report support and maintainability. None of these block the flight repo integration.

| Item | Rationale |
|---|---|
| Flesh out `docs/MODEL_ARCHITECTURE.md` | Currently a stub; needed for capstone report and flight repo developers |
| Add mypy type checking | Deferred in cleanup spec; important before the public API is consumed by another repo |
| Add FP16 export option | Minimum quantization step for edge deployment; validate ONNX FP16 output stays within tolerance |
| Add Docker/devcontainer | Reproducible environment for capstone graders and future collaborators |

---

## 3. Ordering Constraints

```
v0.3 (train + validate)
  └── v0.4 (export + integration interface)
        └── v0.5 (docs + polish)   ← can partially overlap with v0.4
```

Items within v0.5 (architecture doc, mypy) can begin in parallel with v0.4
but are not on the critical path to flight software integration.

---

## 4. Out of Scope

- MLflow / Weights & Biases (TensorBoard via Lightning is sufficient for the capstone).
- Multi-GPU / distributed training.
- Model ensembling or architecture search.
- Automated retraining pipeline.
