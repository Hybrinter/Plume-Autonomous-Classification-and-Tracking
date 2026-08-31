"""tools.inference -- train, export, accept, and score frozen plume-model artifacts.

This package is the single home for model engineering under tools/. SIL telemetry
analysis stays in tools.analysis. Torch and torchvision are required tools
dependencies. Inference batches are torch tensors.

Contains:
  - accept: frozen ONNX intake gate (hash, I/O, IoU or accuracy, latency).
  - metrics: torch classifier and segmentor design metrics.
  - data: synthetic scenes, processed packs, and torch Dataset split loaders.
  - split: frozen train/val/test recipe and dataset hash.
  - fetch: Zenodo 4250706 checksums and streamed labeled preprocess.
  - train: plain-torch SGD loop and local run directories.
  - eval / plots / report / runs: held-out scoring, figures, and compare tables.
  - sweep: cartesian search space over the local run catalog.
  - export: ONNX export, manifest write, and promote.
  - arch: pactnet classifier and dilatenet segmentor defaults.
  - __main__: `python -m tools.inference` subcommands.

Satisfies: REQ-AIML-HIGH-004.
"""

from tools.inference.accept import (
    AcceptanceReport,
    ClassifierAcceptanceReport,
    GoldenClassifierScene,
    GoldenScene,
    Manifest,
    accept_artifact,
    accept_classifier_artifact,
    load_manifest,
    onnx_classifier_inference_fn,
    onnx_inference_fn,
)
from tools.inference.metrics import (
    binary_accuracy,
    compute_iou,
    mean_binary_accuracy,
)

__all__ = [
    "AcceptanceReport",
    "ClassifierAcceptanceReport",
    "GoldenClassifierScene",
    "GoldenScene",
    "Manifest",
    "accept_artifact",
    "accept_classifier_artifact",
    "binary_accuracy",
    "compute_iou",
    "load_manifest",
    "mean_binary_accuracy",
    "onnx_classifier_inference_fn",
    "onnx_inference_fn",
]
