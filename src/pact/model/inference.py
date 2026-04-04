"""
pact.model.inference — InferenceEngine for single-frame plume segmentation.

Satisfies: REQ-AIML-COMP-001, REQ-AIML-COMP-002

The InferenceEngine wraps a loaded smp.Unet model and runs forward passes inside the
inference multiprocessing.Process. It must never be constructed in the same process as
storage, telemetry, or comms subsystems (REQ-AIML-COMP-002).

Frozen-dataclass note:
    InferenceEngine is @dataclass(frozen=True) even though it holds a torch.nn.Module,
    which is mutable. This is an intentional exception: the frozen constraint prevents
    accidental field reassignment on the Python level (e.g. self.model = ...), but it
    cannot prevent in-place weight mutation. The contract is enforced by convention and
    code review — model weights must not change after InferenceEngine is constructed.
    See model/CLAUDE.md for the full rationale.
"""

from __future__ import annotations

# stdlib
import time
from dataclasses import dataclass

# third-party
import numpy as np
import torch
import torch.nn as nn

# internal
from pact.types.enums import FaultCode, Ok, Err, Result
from pact.types.messages import InferenceResultMsg, ProcessedFrameMsg
from pact.types.config import InferenceConfig


@dataclass(frozen=True)
class InferenceEngine:
    """Wraps the loaded segmentation model for frame-level inference.

    Satisfies: REQ-AIML-COMP-001, REQ-AIML-COMP-002

    Must NOT be constructed in the same process as storage, telemetry, or comms.
    Construct once at inference process startup; reuse across frames.

    Attributes:
        model:  Loaded smp.Unet (or compatible nn.Module). Weights are fixed at
                construction time and must not be mutated during operation.
                See the frozen-dataclass note in the module docstring.
        config: InferenceConfig from PactConfig (latency budget, model path, etc.).
        device: torch.device to run inference on (e.g. "cuda:0" or "cpu").
    """

    model: nn.Module           # mutable attribute — see frozen-dataclass note above
    config: InferenceConfig
    device: torch.device

    def run(self, frame: ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]:
        """Run inference on one preprocessed frame.

        Processing steps:
        1. Convert frame.tensor (np.ndarray, shape (4, H, W)) to torch.Tensor.
        2. Add batch dimension → (1, 4, H, W).
        3. Move to self.device.
        4. Forward pass through self.model (raw logits).
        5. Apply sigmoid to produce probability map.
        6. Check for NaN / Inf in output → return Err(INFERENCE_NAN).
        7. Check wall-clock time against config.latency_budget_ms → return Err(INFERENCE_TIMEOUT).
        8. Extract blobs via connected-component analysis on thresholded mask.
        9. Build and return Ok(InferenceResultMsg).

        Validates output against REQ-AIML-DATA-004 (no NaN, no Inf) before returning Ok.

        Args:
            frame: Preprocessed frame from the preprocessing pipeline.

        Returns:
            Ok(InferenceResultMsg) on success.
            Err(FaultCode.INFERENCE_NAN) if the sigmoid output contains NaN or Inf.
            Err(FaultCode.INFERENCE_TIMEOUT) if wall-clock time exceeds
            config.latency_budget_ms.
        """
        ...
