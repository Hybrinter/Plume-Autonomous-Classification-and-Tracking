"""Swappable binary plume-presence classifier backends.

ClassifierBackend is the cheap first stage of onboard inference: a single logit
for one plume class. A negative decision skips the segmentor. SIL uses
ScriptedClassifier; flight uses OnnxClassifier over a frozen .onnx artifact.
onnxruntime is imported lazily inside OnnxClassifier.__init__.

Contains:
  - ClassifierDecision: logit plus the boolean gate result.
  - ClassifierBackend: Protocol classify(frame) -> Result[ClassifierDecision, FaultCode].
  - ScriptedClassifier: fixed positive/negative decision for SIL and tests.
  - OnnxClassifier: ONNX session over (1, C, H, W) -> (1, 1) logit.

Satisfies: REQ-AIML-COMP-001.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from flight.libs.messages import ProcessedFrameMsg
from flight.libs.types import Err, FaultCode, Ok, Result
from flight.payload.inference.onnx_session import OnnxInferenceSession, load_onnx_session


@dataclass(frozen=True, slots=True)
class ClassifierDecision:
    """Outcome of one binary presence classification.

    Attributes:
        logit: Raw network logit. The gate is logit >= threshold, which matches
            sigmoid(logit) >= 0.5 when threshold is 0.0.
        positive: True when the frame should proceed to segmentation.
    """

    logit: float
    positive: bool


@runtime_checkable
class ClassifierBackend(Protocol):
    """Onboard presence classifier: one logit per preprocessed frame."""

    def classify(self, frame: ProcessedFrameMsg) -> Result[ClassifierDecision, FaultCode]:
        """Classify a preprocessed frame as plume-present or empty."""
        ...


class ScriptedClassifier:
    """Deterministic classifier for SIL and tests.

    Ignores the frame tensor and returns a configured decision so closed-loop
    pointing stays reproducible. Default is always-positive.
    """

    def __init__(self, positive: bool = True, logit: float = 1.0) -> None:
        """Configure a fixed presence decision.

        Args:
            positive: If True, frames proceed to the segmentor.
            logit: Reported logit. Ignored by the gate; the boolean wins.
        """
        self._positive = positive
        self._logit = logit

    def classify(self, frame: ProcessedFrameMsg) -> Result[ClassifierDecision, FaultCode]:
        """Return the configured decision. The frame tensor is unused.

        Args:
            frame: Preprocessed frame (ignored).

        Returns:
            Ok(ClassifierDecision) with the configured logit and positive flag.
        """
        del frame
        return Ok(ClassifierDecision(logit=self._logit, positive=self._positive))


class OnnxClassifier:
    """ONNX-runtime binary classifier over a frozen artifact (flight).

    Expected I/O: (1, C, H, W) float32 in, (1, 1) logit out. A frame is positive
    when the squeezed logit is >= logit_threshold (default 0.0).
    """

    def __init__(
        self,
        model_path: str,
        logit_threshold: float = 0.0,
        expected_sha256: str | None = None,
        expected_input_shape: tuple[int | None, ...] | None = None,
        expected_output_shape: tuple[int | None, ...] | None = None,
    ) -> None:
        """Open an onnxruntime session over the classifier artifact.

        Args:
            model_path: Filesystem path to the frozen classifier .onnx.
            logit_threshold: Positive when logit >= this value. Default 0.0.
            expected_sha256: Optional SHA-256 checked before session creation.
            expected_input_shape: Optional input shape checked after load.
            expected_output_shape: Optional output shape checked after load.

        Raises:
            ImportError: If onnxruntime is not installed.
            ValueError: If hash or I/O contract verification fails.
        """
        self._session: OnnxInferenceSession = load_onnx_session(
            model_path,
            expected_sha256=expected_sha256,
            expected_input_shape=expected_input_shape,
            expected_output_shape=expected_output_shape,
        )
        self._logit_threshold = logit_threshold

    def classify(self, frame: ProcessedFrameMsg) -> Result[ClassifierDecision, FaultCode]:
        """Run the classifier session and gate on the logit threshold.

        Args:
            frame: Preprocessed frame whose tensor is (C, H, W) float32.

        Returns:
            Ok(ClassifierDecision) on a finite logit, else Err(INFERENCE_NAN).
        """
        bands = np.asarray(frame.tensor, dtype=np.float32)  # np.ndarray[float32, (C, H, W)]
        model_input = bands[np.newaxis, ...]  # (1, C, H, W)
        input_name = self._session.get_inputs()[0].name
        raw = self._session.run(None, {input_name: model_input})[0]
        logit = float(np.asarray(raw, dtype=np.float32).reshape(-1)[0])
        if not np.isfinite(logit):
            return Err(FaultCode.INFERENCE_NAN)
        return Ok(ClassifierDecision(logit=logit, positive=logit >= self._logit_threshold))
