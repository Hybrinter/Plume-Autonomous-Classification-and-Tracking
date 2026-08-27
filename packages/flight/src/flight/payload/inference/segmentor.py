"""Swappable segmentation backends that emit a per-pixel probability mask.

SegmentorBackend is the second onboard inference stage. It runs only after the
classifier reports plume-present. SIL uses ScriptedSegmentor; flight uses
OnnxSegmentor over a frozen .onnx artifact. onnxruntime is imported lazily
inside session load.

Contains:
  - SegmentorBackend: Protocol segment(frame) -> Result[ndarray, FaultCode].
  - ScriptedSegmentor: fixed (H, W) probability mask for SIL and tests.
  - OnnxSegmentor: ONNX session over (1, C, H, W) -> (1, 1, H, W) logits, then sigmoid.

Satisfies: REQ-AIML-COMP-001.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from flight.libs.messages import ProcessedFrameMsg
from flight.libs.types import Err, FaultCode, Ok, Result
from flight.payload.inference.onnx_session import OnnxInferenceSession, load_onnx_session


@runtime_checkable
class SegmentorBackend(Protocol):
    """Onboard segmentor: per-pixel plume probabilities from a preprocessed frame."""

    def segment(self, frame: ProcessedFrameMsg) -> Result[np.ndarray, FaultCode]:
        """Segment a preprocessed frame into a probability mask."""
        ...


class ScriptedSegmentor:
    """Deterministic segmentor backed by a fixed probability mask (SIL/tests).

    Ignores the frame tensor. The mask is returned as-is so extract_blobs sees
    the same geometry as a live ONNX path.
    """

    def __init__(self, prob_mask: np.ndarray) -> None:
        """Configure the fixed probability mask.

        Args:
            prob_mask: (H, W) float32 probabilities in [0, 1].
        """
        self._prob_mask = np.asarray(prob_mask, dtype=np.float32)  # np.ndarray[float32, (H, W)]

    def segment(self, frame: ProcessedFrameMsg) -> Result[np.ndarray, FaultCode]:
        """Return the configured mask. The frame tensor is unused.

        Args:
            frame: Preprocessed frame (ignored).

        Returns:
            Ok(np.ndarray[float32, (H, W)]) of the configured mask.
        """
        del frame
        return Ok(self._prob_mask)


class OnnxSegmentor:
    """ONNX-runtime segmentor over a frozen artifact (flight).

    Expected I/O: (1, C, H, W) float32 in, (1, 1, H, W) logits out. Sigmoid is
    applied in this module; the exported graph must not bake sigmoid in.
    """

    def __init__(
        self,
        model_path: str,
        expected_sha256: str | None = None,
        expected_input_shape: tuple[int | None, ...] | None = None,
        expected_output_shape: tuple[int | None, ...] | None = None,
    ) -> None:
        """Open an onnxruntime session over the segmentor artifact.

        Args:
            model_path: Filesystem path to the frozen segmentor .onnx.
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

    def segment(self, frame: ProcessedFrameMsg) -> Result[np.ndarray, FaultCode]:
        """Run the session, apply sigmoid, and return the (H, W) probability mask.

        Args:
            frame: Preprocessed frame whose tensor is (C, H, W) float32.

        Returns:
            Ok(np.ndarray[float32, (H, W)]) on finite output, else Err(INFERENCE_NAN).
        """
        bands = np.asarray(frame.tensor, dtype=np.float32)  # np.ndarray[float32, (C, H, W)]
        model_input = bands[np.newaxis, ...]  # (1, C, H, W)
        input_name = self._session.get_inputs()[0].name
        logits = self._session.run(None, {input_name: model_input})[0]
        probs = 1.0 / (1.0 + np.exp(-logits))
        if not bool(np.isfinite(probs).all()):
            return Err(FaultCode.INFERENCE_NAN)
        return Ok(probs[0, 0].astype(np.float32))  # np.ndarray[float32, (H, W)]
