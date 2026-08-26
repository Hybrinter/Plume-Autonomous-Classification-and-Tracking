"""Compatibility re-export of the model acceptance gate.

New code imports `tools.model.accept`. This module forwards the same names.
"""

from tools.model.accept import (
    AcceptanceReport,
    GoldenScene,
    InferenceFn,
    Manifest,
    Shape,
    accept_artifact,
    compute_iou,
    load_manifest,
    onnx_inference_fn,
)

__all__ = [
    "AcceptanceReport",
    "GoldenScene",
    "InferenceFn",
    "Manifest",
    "Shape",
    "accept_artifact",
    "compute_iou",
    "load_manifest",
    "onnx_inference_fn",
]
