"""Swappable detection backends for the payload.

DetectorBackend abstracts onboard detection: SIL/tests use the deterministic
ScriptedDetector while flight uses OnnxDetector (a frozen ONNX artifact run via
onnxruntime). onnxruntime is imported lazily in OnnxDetector.__init__, so importing
this module never requires it -- mirroring the camera SDK pattern. Both backends
share extract_blobs for identical detection geometry.
"""

import time
from typing import Protocol, runtime_checkable

import numpy as np

from flight.libs.messages import InferenceResultMsg, ProcessedFrameMsg
from flight.libs.types import Err, FaultCode, MessageType, Ok, Result
from flight.payload.model.blobs import extract_blobs
from flight.payload.model.verify import (
    check_inference_latency,
    verify_io_contract,
    verify_model_hash,
)


@runtime_checkable
class DetectorBackend(Protocol):
    """Onboard detector: turns a preprocessed frame into a detection result."""

    def detect(self, frame: ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]:
        """Run detection on a preprocessed frame."""
        ...


class ScriptedDetector:
    """Deterministic detector backed by a fixed probability mask (SIL/tests).

    Each detect() runs the shared extract_blobs over the configured mask, exercising
    the real detection geometry without a model.
    """

    def __init__(
        self,
        prob_mask: np.ndarray,
        confidence_gate: float = 0.55,
        min_blob_area_px: int = 15,
        model_version: str = "scripted",
    ) -> None:
        """Configure the scripted mask and detection thresholds."""
        self._prob_mask = prob_mask
        self._confidence_gate = confidence_gate
        self._min_blob_area_px = min_blob_area_px
        self._model_version = model_version

    def detect(self, frame: ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]:
        """Return a detection result built from the fixed mask for this frame."""
        blobs = extract_blobs(self._prob_mask, self._confidence_gate, self._min_blob_area_px)
        return Ok(
            InferenceResultMsg(
                msg_type=MessageType.INFERENCE_RESULT,
                timestamp_utc=frame.timestamp_utc,
                frame_id=frame.frame_id,
                mask=self._prob_mask,
                blobs=blobs,
                model_version=self._model_version,
                inference_ms=0.0,
                mode_flags=0,
                crop_origin_px=frame.crop_origin_px,
                scale_factor=frame.scale_factor,
            )
        )


class OnnxDetector:
    """ONNX-runtime detector over a frozen model artifact (flight).

    onnxruntime is imported lazily in __init__; importing this module does not
    require it. detect() runs the session, applies sigmoid + threshold, and reuses
    extract_blobs. Exercised only when onnxruntime and a real .onnx artifact exist.
    """

    def __init__(
        self,
        model_path: str,
        confidence_gate: float = 0.55,
        min_blob_area_px: int = 15,
        model_version: str = "unknown",
        expected_sha256: str | None = None,
        latency_budget_ms: float = 0.0,
        expected_input_shape: tuple[int | None, ...] | None = None,
        expected_output_shape: tuple[int | None, ...] | None = None,
    ) -> None:
        """Open an onnxruntime session over model_path, verifying hash + I/O contract at load.

        If expected_sha256 is given the artifact's SHA-256 is verified BEFORE the session is
        created (a corrupt artifact is rejected without loading it); if the expected I/O shapes
        are given the session's input/output shapes are verified after load. A verification
        failure raises ValueError -- a corrupt/incompatible model at startup is unrecoverable
        (the composition-root startup-exception contract), conceptually a MODEL_CORRUPT fault.
        latency_budget_ms (> 0) caps per-frame inference time in detect (INFERENCE_TIMEOUT).

        Raises:
            ImportError: If onnxruntime is not installed.
            ValueError: If the artifact hash or I/O contract verification fails.
        """
        if expected_sha256 is not None:
            hash_result = verify_model_hash(model_path, expected_sha256)
            if isinstance(hash_result, Err):
                raise ValueError(f"model hash verification failed ({hash_result.error.value})")
        try:
            import onnxruntime
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is not installed. Install it and provide a frozen .onnx "
                "artifact to use OnnxDetector; use ScriptedDetector in tests and simulation."
            ) from exc
        self._session = onnxruntime.InferenceSession(model_path)
        self._confidence_gate = confidence_gate
        self._min_blob_area_px = min_blob_area_px
        self._model_version = model_version
        self._latency_budget_ms = latency_budget_ms
        if expected_input_shape is not None and expected_output_shape is not None:
            actual_in = self._onnx_shape(self._session.get_inputs()[0].shape)
            actual_out = self._onnx_shape(self._session.get_outputs()[0].shape)
            contract = verify_io_contract(
                actual_in, actual_out, expected_input_shape, expected_output_shape
            )
            if isinstance(contract, Err):
                raise ValueError(f"model I/O contract verification failed ({contract.error.value})")

    @staticmethod
    def _onnx_shape(shape: list[object]) -> tuple[int | None, ...]:
        """Normalize an onnxruntime tensor shape (symbolic dims -> None) to a typed tuple."""
        return tuple(dim if isinstance(dim, int) else None for dim in shape)

    def detect(self, frame: ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]:
        """Run the ONNX session on the frame tensor and extract blobs (latency-budgeted)."""
        start = time.perf_counter()
        bands = np.asarray(frame.tensor, dtype=np.float32)  # np.ndarray[float32, (C, H, W)]
        model_input = bands[np.newaxis, ...]  # (1, C, H, W)
        input_name = self._session.get_inputs()[0].name
        logits = self._session.run(None, {input_name: model_input})[0]  # (1, 1, H, W)
        probs = 1.0 / (1.0 + np.exp(-logits))
        if not bool(np.isfinite(probs).all()):
            return Err(FaultCode.INFERENCE_NAN)
        prob_mask = probs[0, 0].astype(np.float32)  # (H, W)
        blobs = extract_blobs(prob_mask, self._confidence_gate, self._min_blob_area_px)
        inference_ms = (time.perf_counter() - start) * 1000.0
        latency = check_inference_latency(inference_ms, self._latency_budget_ms)
        if isinstance(latency, Err):
            return Err(latency.error)
        return Ok(
            InferenceResultMsg(
                msg_type=MessageType.INFERENCE_RESULT,
                timestamp_utc=frame.timestamp_utc,
                frame_id=frame.frame_id,
                mask=prob_mask,
                blobs=blobs,
                model_version=self._model_version,
                inference_ms=inference_ms,
                mode_flags=0,
                crop_origin_px=frame.crop_origin_px,
                scale_factor=frame.scale_factor,
            )
        )
