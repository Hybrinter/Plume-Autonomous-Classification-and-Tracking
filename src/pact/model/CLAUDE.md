# model/ — PACT Subsystem Context

## Purpose
U-Net/ResNet-34 segmentation model definition, training, evaluation, and inference for
multispectral VNIR plume detection.

## Satisfies
- REQ-AIML-HIGH-001 — onboard plume segmentation from VNIR imagery
- REQ-AIML-HIGH-002 — pretrained ResNet-34 encoder with ImageNet weights
- REQ-AIML-IMAG-001 — 4-band (B2/B3/B4/B8) input tensor format
- REQ-AIML-COMP-001 — inference runs in an isolated multiprocessing.Process
- REQ-AIML-COMP-002 — inference process is isolated from storage, telemetry, and comms

## Owns
- `InferenceResultMsg` — produced by `inference.py` after each forward pass

## Consumes
- `ProcessedFrameMsg` — receives from the preprocessing pipeline (runs inside this process)

## Key Invariants
- `InferenceEngine` MUST NOT be constructed in the same process as storage, telemetry, or
  comms subsystems (REQ-AIML-COMP-002). Violating this breaks GPU memory isolation on Jetson.
- Model weights do not change during inference. The frozen dataclass pattern enforces this
  structurally; see the note in `inference.py` about the mutable-module exception.
- Preprocessing runs inside the inference process as a plain function call — not a separate
  process or thread. See `preprocessing/adr/ADR-001` for the rationale.

## Concurrency
`multiprocessing.Process` with `multiprocessing.Queue` (see `model/adr/ADR-001`).
Rationale: GPU workloads require true process isolation from the GIL; `multiprocessing`
provides this and satisfies REQ-AIML-COMP-002.

## Known Gaps / TODOs
- TensorRT INT8 calibration is not implemented — `quantize.py` is a stub only.
  See `# TODO: replace with TensorRT INT8 calibration for Jetson Xavier deployment`.
- Actual model weights are not trained. `data/models/active.pt` does not exist until
  training is run against the HSG-AIML dataset (see `dataset.py` and `train.py`).
- `latency_budget_ms = 500.0` in `InferenceConfig` is a placeholder pending a real
  Jetson Xavier NX/AGX benchmark with the trained model.
- `download_dataset()` in `dataset.py` is a stub — Zenodo fetch not yet implemented.
