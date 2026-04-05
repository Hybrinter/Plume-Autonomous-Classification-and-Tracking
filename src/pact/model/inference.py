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
import scipy.ndimage
import torch
import torch.nn as nn

# internal
from pact.types.enums import FaultCode, MessageType, Ok, Err, Result
from pact.types.messages import BlobMeta, InferenceResultMsg, ProcessedFrameMsg
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
        confidence_gate: Threshold for binary mask (default 0.55).
        min_blob_area_px: Minimum blob pixel area to include (default 15).
        model_version: Model checkpoint identifier string.
    """

    model: nn.Module           # mutable attribute — see frozen-dataclass note above
    config: InferenceConfig
    device: torch.device
    confidence_gate: float = 0.55
    min_blob_area_px: int = 15
    model_version: str = "unknown"

    def run(self, frame: ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]:
        """Run inference on one preprocessed frame.

        Processing steps:
        1. Convert frame.tensor (np.ndarray, shape (4, H, W)) to torch.Tensor.
        2. Add batch dimension → (1, 4, H, W).
        3. Move to self.device.
        4. Forward pass through self.model (raw logits).
        5. Apply sigmoid to produce probability map.
        6. Check for NaN / Inf in output → return Err(INFERENCE_NAN).
        7. Check wall-clock time against config.latency_budget_ms
           → return Err(INFERENCE_TIMEOUT).
        8. Extract blobs via connected-component analysis on thresholded mask.
        9. Build and return Ok(InferenceResultMsg).

        Validates output against REQ-AIML-DATA-004 (no NaN, no Inf)
        before returning Ok.

        Args:
            frame: Preprocessed frame from the preprocessing pipeline.

        Returns:
            Ok(InferenceResultMsg) on success.
            Err(FaultCode.INFERENCE_NAN) if the sigmoid output contains
            NaN or Inf.
            Err(FaultCode.INFERENCE_TIMEOUT) if wall-clock time exceeds
            config.latency_budget_ms.
        """
        t_start = time.perf_counter()

        # --- 1. Validate input shape ---
        bands: np.ndarray = frame.tensor  # type: ignore[assignment]
        if bands.ndim != 3 or bands.shape[0] != 4:
            return Err(FaultCode.INFERENCE_NAN)

        # --- 2-3. Convert to tensor, add batch dim, move to device ---
        tensor = torch.from_numpy(
            bands.astype(np.float32)
        ).unsqueeze(0).to(self.device)  # (1, 4, H, W)

        # --- 4-5. Forward pass + sigmoid ---
        with torch.no_grad():
            logits = self.model(tensor)  # (1, 1, H, W)
        probs = torch.sigmoid(logits)

        # --- 6. NaN / Inf check ---
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            return Err(FaultCode.INFERENCE_NAN)

        # --- 7. Latency budget check ---
        t_end = time.perf_counter()
        inference_ms = (t_end - t_start) * 1000.0
        if inference_ms > self.config.latency_budget_ms:
            return Err(FaultCode.INFERENCE_TIMEOUT)

        # --- 8. Blob extraction via connected components ---
        prob_np = probs[0, 0].cpu().numpy()  # (H, W) float32
        binary_mask = (
            prob_np >= self.confidence_gate
        ).astype(np.uint8)

        labeled, num_features = scipy.ndimage.label(binary_mask)

        blobs: list[BlobMeta] = []
        for label_idx in range(1, num_features + 1):
            component = labeled == label_idx
            pixel_area = int(component.sum())
            if pixel_area < self.min_blob_area_px:
                continue
            ys, xs = np.where(component)
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            cx = float(xs.mean())
            cy = float(ys.mean())
            mean_conf = float(prob_np[component].mean())
            blobs.append(BlobMeta(
                blob_id=0,
                bbox=(x_min, y_min, x_max, y_max),
                centroid_raw=(cx, cy),
                pixel_area=pixel_area,
                mean_confidence=mean_conf,
                persistence_count=0,
            ))

        # --- 9. Build result ---
        mode_flags = 0
        return Ok(InferenceResultMsg(
            msg_type=MessageType.INFERENCE_RESULT,
            timestamp_utc=frame.timestamp_utc,
            frame_id=frame.frame_id,
            mask=prob_np,
            blobs=tuple(blobs),
            model_version=self.model_version,
            inference_ms=inference_ms,
            mode_flags=mode_flags,
        ))
