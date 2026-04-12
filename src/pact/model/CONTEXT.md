# model/ -- Agent Context

## Purpose

Runs the U-Net/ResNet-34 segmentation forward pass and extracts connected-component blobs
from the post-sigmoid confidence mask. Also contains ground-side training, evaluation, and
dataset tools -- these are never imported in flight code.

## Defining Design Decision

`InferenceEngine` is `@dataclass(frozen=True)` but holds a mutable `torch.nn.Module`.
The frozen constraint prevents field *reassignment* -- it cannot prevent in-place weight
mutation. Weights must not change after construction. This is enforced by convention only.

## Invariants

`InferenceEngine` must never be constructed in the same OS process as storage, telemetry,
or comms (REQ-AIML-COMP-002). Violating this breaks GPU memory isolation on the Jetson.
The only legal construction site is inside `_run_inference_process()` in `ops/main.py`.

## Gotchas

Blob extraction uses `scipy.ndimage.label` on the *post-sigmoid, post-threshold* mask --
not on raw logits. The `confidence_gate` threshold is applied to sigmoid probabilities
[0,1] before connected-component analysis. A gate of 0.55 means only pixels with
predicted plume probability > 55% are candidates for blob regions.

## Phase II Gaps

- `quantize.py` (TensorRT INT8) is a stub -- not implemented.
- `latency_budget_ms = 500.0` is a placeholder pending a Jetson Xavier benchmark with
  the trained model. The timeout fault threshold is currently unvalidated.
- `data/models/active.pt` does not exist until training runs against the HSG-AIML dataset.
