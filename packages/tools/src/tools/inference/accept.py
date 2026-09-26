"""Model-artifact acceptance gate.

The implementation lives in ``tools.ml_models.export.accept``. This module
re-exports that public surface so existing callers keep one import path.

Contains:
  - Manifest / GoldenScene / GoldenClassifierScene / AcceptanceReport.
  - ClassifierAcceptanceReport: classifier-specific quality fields.
  - load_manifest / accept_artifact / accept_classifier_artifact / accept_kind.
  - load_golden_scenes / load_golden_classifier_scenes.
  - compute_iou: re-export from tools.inference.metrics.
  - onnx_inference_fn / onnx_classifier_inference_fn.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from tools.ml_models.export.accept import (
    AcceptanceReport,
    ClassifierAcceptanceReport,
    GoldenClassifierScene,
    GoldenScene,
    Manifest,
    accept_artifact,
    accept_classifier_artifact,
    accept_kind,
    compute_iou,
    load_golden_classifier_scenes,
    load_golden_scenes,
    load_manifest,
    onnx_classifier_inference_fn,
    onnx_inference_fn,
)
from tools.ml_models.export.accept import _latency_stats as _latency_stats

__all__ = [
    "AcceptanceReport",
    "ClassifierAcceptanceReport",
    "GoldenClassifierScene",
    "GoldenScene",
    "Manifest",
    "accept_artifact",
    "accept_classifier_artifact",
    "accept_kind",
    "compute_iou",
    "load_golden_classifier_scenes",
    "load_golden_scenes",
    "load_manifest",
    "onnx_classifier_inference_fn",
    "onnx_inference_fn",
]
