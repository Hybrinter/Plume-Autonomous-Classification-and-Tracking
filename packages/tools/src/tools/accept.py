"""Compatibility re-export of the model acceptance gate.

New code imports `tools.model.accept`. This module forwards the same names.
"""

from tools.model.accept import (
    AcceptanceReport,
    ClassifierAcceptanceReport,
    ClassifierInferenceFn,
    GoldenClassifierScene,
    GoldenScene,
    InferenceFn,
    Manifest,
    Shape,
    accept_artifact,
    accept_classifier_artifact,
    compute_iou,
    load_manifest,
    onnx_classifier_inference_fn,
    onnx_inference_fn,
)

__all__ = [
    "AcceptanceReport",
    "ClassifierAcceptanceReport",
    "ClassifierInferenceFn",
    "GoldenClassifierScene",
    "GoldenScene",
    "InferenceFn",
    "Manifest",
    "Shape",
    "accept_artifact",
    "accept_classifier_artifact",
    "compute_iou",
    "load_manifest",
    "onnx_classifier_inference_fn",
    "onnx_inference_fn",
]
