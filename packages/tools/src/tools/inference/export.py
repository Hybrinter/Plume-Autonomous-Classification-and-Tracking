"""Torch-to-ONNX export, manifest sidecar, FP16/INT8 conversion, and promote.

The implementation lives in ``tools.ml_models.export.onnx``. This module
re-exports that public surface so existing callers keep one import path.

Contains:
  - ExportConfig: frozen export hyperparameters.
  - export: write one ONNX graph plus a Manifest JSON sidecar.
  - reexport_spatial: rebuild a graph at a new H/W and copy same-name weights.
    A same-name shape mismatch is an error.
  - convert_fp16: rewrite an FP32 graph to FP16 with float32 I/O.
  - quantize_int8: static QDQ INT8 with float32 I/O.
  - quantize_knee: classifier FP16 and segmentor INT8 in place.
  - write_manifest: serialize a Manifest.
  - int8_artifact_path: sibling ``*.int8.onnx`` path for an FP32 artifact.
  - fp16_artifact_path: sibling ``*.fp16.onnx`` path for an FP32 artifact.
  - promote: copy a passed artifact into the destination path.
  - resolve_export_hw: choose the traced H/W for a checkpoint.
  - GateReport: protocol with accepted + detail.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from tools.ml_models.export.onnx import (
    ExportConfig,
    GateReport,
    convert_fp16,
    export,
    fp16_artifact_path,
    int8_artifact_path,
    promote,
    quantize_int8,
    quantize_knee,
    reexport_spatial,
    resolve_export_hw,
    write_manifest,
)
from tools.ml_models.export.onnx import _calibration_batches as _calibration_batches
from tools.ml_models.export.onnx import (
    _copy_matching_initializers as _copy_matching_initializers,
)

__all__ = [
    "ExportConfig",
    "GateReport",
    "convert_fp16",
    "export",
    "fp16_artifact_path",
    "int8_artifact_path",
    "promote",
    "quantize_int8",
    "quantize_knee",
    "reexport_spatial",
    "resolve_export_hw",
    "write_manifest",
]
