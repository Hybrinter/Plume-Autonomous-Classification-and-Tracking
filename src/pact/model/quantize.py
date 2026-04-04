"""
pact.model.quantize — INT8 quantisation for Jetson Xavier deployment.

Satisfies: REQ-AIML-COMP-001 (inference latency budget)

Current implementation uses torch.quantization dynamic quantisation, which is a
pure-Python stub. This must be replaced with TensorRT INT8 calibration before any
Jetson Xavier flight deployment — see the TODO annotation on quantize_to_int8().

Known gap: TensorRT INT8 calibration not implemented (stub only).
See model/CLAUDE.md → Known Gaps for the full list.
"""

from __future__ import annotations

# stdlib
from typing import Any

# third-party
import torch
import torch.nn as nn


def quantize_to_int8(
    model_path: str,
    output_path: str,
    calibration_loader: Any,  # DataLoader yielding calibration samples
) -> None:
    """Quantise a saved model to INT8 and write the result to output_path.

    Current implementation applies PyTorch dynamic quantisation as a placeholder.

    # TODO: replace with TensorRT INT8 calibration for Jetson Xavier deployment
    #
    # The production implementation should:
    #   1. Load the FP32 model from model_path.
    #   2. Run calibration_loader samples through a TensorRT calibrator (IInt8Calibrator).
    #   3. Build a TensorRT engine with INT8 precision enabled.
    #   4. Serialise the engine to output_path as a .trt plan file.
    #   5. Verify that the quantised engine produces outputs within an acceptable tolerance
    #      of the FP32 model on a held-out validation set.
    #
    # Reference: TensorRT Developer Guide §4 (INT8 Inference).
    # Target: Jetson Xavier NX / AGX, TensorRT >= 8.5.

    Args:
        model_path:        Path to the source FP32 model (.pt checkpoint).
        output_path:       Destination path for the quantised model artefact.
        calibration_loader: DataLoader yielding representative calibration batches.
            Each batch should be a (tensor,) tuple with shape (N, 4, 256, 256) float32.
            Ignored by the current dynamic-quantisation stub.
    """
    # Stub: dynamic quantisation (CPU only, no calibration data used).
    model: nn.Module = torch.load(model_path, map_location="cpu")
    quantized_model: nn.Module = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear},
        dtype=torch.qint8,
    )
    torch.save(quantized_model, output_path)
