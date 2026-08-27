"""Swappable detection backends: classifier, then maybe segmentor, then blobs.

DetectorBackend remains the single entry the payload app calls. Each detect()
runs ClassifierBackend first. A negative presence decision skips the segmentor
and returns an empty blob list with a zero mask so control continues in search.
A positive decision runs SegmentorBackend, then extract_blobs.

SIL/tests use ScriptedDetector (always-positive classifier by default plus a
fixed mask). Flight uses OnnxDetector, which loads two frozen .onnx artifacts.
onnxruntime is imported lazily inside session load.

Contains:
  - Detector: composer of classifier + segmentor + blob extract.
  - DetectorBackend: Protocol detect(frame) -> Result[InferenceResultMsg, FaultCode].
  - ScriptedDetector: ScriptedClassifier + ScriptedSegmentor convenience.
  - OnnxDetector: OnnxClassifier + OnnxSegmentor convenience.

Satisfies: REQ-AIML-COMP-001.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

import numpy as np

from flight.libs.messages import BlobMeta, InferenceResultMsg, ProcessedFrameMsg
from flight.libs.types import Err, FaultCode, MessageType, Ok, Result
from flight.payload.blobs import extract_blobs
from flight.payload.inference.classifier import (
    ClassifierBackend,
    OnnxClassifier,
    ScriptedClassifier,
)
from flight.payload.inference.segmentor import OnnxSegmentor, ScriptedSegmentor, SegmentorBackend
from flight.payload.inference.verify import check_inference_latency


@runtime_checkable
class DetectorBackend(Protocol):
    """Onboard detector: turns a preprocessed frame into a detection result."""

    def detect(self, frame: ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]:
        """Run detection on a preprocessed frame."""
        ...


class Detector:
    """Compose classifier then segmentor then blob extraction.

    Attributes are the injected backends and blob thresholds. Latency is measured
    over the full detect() call, including a negative-class skip.
    """

    def __init__(
        self,
        classifier: ClassifierBackend,
        segmentor: SegmentorBackend,
        confidence_gate: float = 0.55,
        min_blob_area_px: int = 15,
        model_version: str = "unknown",
        latency_budget_ms: float = 0.0,
        record_wall_clock: bool = True,
    ) -> None:
        """Bind backends and blob/latency thresholds.

        Args:
            classifier: Presence gate. Negative skips segmentation.
            segmentor: Probability-mask producer for positive frames.
            confidence_gate: Blob mean-confidence threshold.
            min_blob_area_px: Minimum connected-component area.
            model_version: Identifier copied onto InferenceResultMsg.
            latency_budget_ms: Wall-clock budget for the whole detect() call.
                Values <= 0 disable the check.
            record_wall_clock: If False, report inference_ms as 0.0 (scripted SIL).
        """
        self._classifier = classifier
        self._segmentor = segmentor
        self._confidence_gate = confidence_gate
        self._min_blob_area_px = min_blob_area_px
        self._model_version = model_version
        self._latency_budget_ms = latency_budget_ms
        self._record_wall_clock = record_wall_clock

    def detect(self, frame: ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]:
        """Classify, optionally segment, extract blobs, and enforce the latency budget.

        Args:
            frame: Preprocessed frame with tensor (C, H, W) float32.

        Returns:
            Ok(InferenceResultMsg) with blobs from the mask, or empty blobs when
            the classifier is negative. Err on classifier/segmentor fault or
            INFERENCE_TIMEOUT.

        Notes:
            A negative classifier returns a zero mask of the tensor spatial size
            and does not call the segmentor.
        """
        start = time.perf_counter() if self._record_wall_clock else 0.0
        decision = self._classifier.classify(frame)
        if isinstance(decision, Err):
            return decision
        blobs: tuple[BlobMeta, ...]
        if not decision.value.positive:
            height = int(np.asarray(frame.tensor).shape[1])
            width = int(np.asarray(frame.tensor).shape[2])
            prob_mask = np.zeros((height, width), dtype=np.float32)  # np.ndarray[float32, (H, W)]
            blobs = ()
        else:
            mask_result = self._segmentor.segment(frame)
            if isinstance(mask_result, Err):
                return mask_result
            prob_mask = mask_result.value  # np.ndarray[float32, (H, W)]
            blobs = extract_blobs(prob_mask, self._confidence_gate, self._min_blob_area_px)
        inference_ms = (time.perf_counter() - start) * 1000.0 if self._record_wall_clock else 0.0
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


class ScriptedDetector(Detector):
    """Deterministic detector for SIL and tests.

    Default classifier is always-positive so existing pointing tests keep a
    stable blob. Pass classifier_positive=False to exercise the skip path.
    """

    def __init__(
        self,
        prob_mask: np.ndarray,
        confidence_gate: float = 0.55,
        min_blob_area_px: int = 15,
        model_version: str = "scripted",
        classifier_positive: bool = True,
        latency_budget_ms: float = 0.0,
    ) -> None:
        """Configure a scripted mask and optional always-negative classifier.

        Args:
            prob_mask: (H, W) float32 probabilities used when the classifier is
                positive.
            confidence_gate: Blob mean-confidence threshold.
            min_blob_area_px: Minimum connected-component area.
            model_version: Identifier copied onto InferenceResultMsg.
            classifier_positive: If False, skip the segmentor every frame.
            latency_budget_ms: Wall-clock budget for detect(); 0 disables it.
        """
        super().__init__(
            classifier=ScriptedClassifier(positive=classifier_positive),
            segmentor=ScriptedSegmentor(prob_mask),
            confidence_gate=confidence_gate,
            min_blob_area_px=min_blob_area_px,
            model_version=model_version,
            latency_budget_ms=latency_budget_ms,
            record_wall_clock=False,
        )


class OnnxDetector(Detector):
    """ONNX-runtime detector over two frozen artifacts (flight).

    Loads a classifier graph and a segmentor graph. The payload app still calls
    detect() once per frame.
    """

    def __init__(
        self,
        segmentor_model_path: str,
        classifier_model_path: str,
        confidence_gate: float = 0.55,
        min_blob_area_px: int = 15,
        model_version: str = "unknown",
        logit_threshold: float = 0.0,
        classifier_sha256: str | None = None,
        segmentor_sha256: str | None = None,
        latency_budget_ms: float = 0.0,
        expected_input_shape: tuple[int | None, ...] | None = None,
        expected_segmentor_output_shape: tuple[int | None, ...] | None = None,
        expected_classifier_output_shape: tuple[int | None, ...] | None = None,
    ) -> None:
        """Open classifier and segmentor sessions, then compose them.

        Args:
            segmentor_model_path: Path to the frozen segmentor .onnx.
            classifier_model_path: Path to the frozen classifier .onnx.
            confidence_gate: Blob mean-confidence threshold.
            min_blob_area_px: Minimum connected-component area.
            model_version: Identifier copied onto InferenceResultMsg.
            logit_threshold: Classifier positive gate (logit >= threshold).
            classifier_sha256: Optional classifier artifact digest.
            segmentor_sha256: Optional segmentor artifact digest.
            latency_budget_ms: Wall-clock budget for the full detect() call.
            expected_input_shape: Optional shared input shape (1, C, H, W).
            expected_segmentor_output_shape: Optional (1, 1, H, W) logits.
            expected_classifier_output_shape: Optional (1, 1) logit.

        Raises:
            ImportError: If onnxruntime is not installed.
            ValueError: If hash or I/O contract verification fails.
        """
        classifier = OnnxClassifier(
            classifier_model_path,
            logit_threshold=logit_threshold,
            expected_sha256=classifier_sha256,
            expected_input_shape=expected_input_shape,
            expected_output_shape=expected_classifier_output_shape,
        )
        segmentor = OnnxSegmentor(
            segmentor_model_path,
            expected_sha256=segmentor_sha256,
            expected_input_shape=expected_input_shape,
            expected_output_shape=expected_segmentor_output_shape,
        )
        super().__init__(
            classifier=classifier,
            segmentor=segmentor,
            confidence_gate=confidence_gate,
            min_blob_area_px=min_blob_area_px,
            model_version=model_version,
            latency_budget_ms=latency_budget_ms,
        )
