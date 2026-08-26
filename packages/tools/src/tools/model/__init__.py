"""tools.model -- train, export, accept, and score frozen plume-model artifacts.

This package is the single home for model engineering under tools/. SIL telemetry
analysis stays in tools.analysis. Torch imports are lazy and live behind the
train extra; this package imports without torch.

Contains:
  - accept: frozen ONNX intake gate (hash, I/O, IoU, latency).
  - metrics: classifier accuracy and mask IoU helpers.
  - train: scaffold; the train loop is not implemented in this layer.
  - export: scaffold; ONNX export is not implemented in this layer.
  - __main__: `python -m tools.model` subcommands.

Satisfies: REQ-AIML-HIGH-004.
"""

from tools.model.accept import (
    AcceptanceReport,
    GoldenScene,
    Manifest,
    accept_artifact,
    compute_iou,
    load_manifest,
    onnx_inference_fn,
)
from tools.model.metrics import binary_accuracy, mean_binary_accuracy

__all__ = [
    "AcceptanceReport",
    "GoldenScene",
    "Manifest",
    "accept_artifact",
    "binary_accuracy",
    "compute_iou",
    "load_manifest",
    "mean_binary_accuracy",
    "onnx_inference_fn",
]
