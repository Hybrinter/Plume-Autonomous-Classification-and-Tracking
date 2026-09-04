"""Lazy onnxruntime session loader shared by the classifier and segmentor.

onnxruntime is imported inside load_onnx_session, so importing this module never
requires the SDK. Hash verification runs before the session is created. Shape
verification runs after load when both expected shapes are given.

Contains:
  - onnx_tensor_shape: normalize an onnxruntime dim list to a typed tuple.
  - load_onnx_session: open a session with optional hash, I/O, and provider list.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

import numpy as np

from flight.libs.types import Err
from flight.payload.inference.verify import verify_io_contract, verify_model_hash


class OnnxNamedValue(Protocol):
    """Subset of onnxruntime NodeArg used after session load."""

    name: str
    shape: list[object]


class OnnxInferenceSession(Protocol):
    """Subset of onnxruntime.InferenceSession used by the ONNX backends."""

    def get_inputs(self) -> Sequence[OnnxNamedValue]:
        """Return input metadata."""
        ...

    def get_outputs(self) -> Sequence[OnnxNamedValue]:
        """Return output metadata."""
        ...

    def run(
        self, output_names: list[str] | None, input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        """Run the graph and return output tensors."""
        ...


def onnx_tensor_shape(shape: list[object]) -> tuple[int | None, ...]:
    """Normalize an onnxruntime tensor shape (symbolic dims become None).

    Args:
        shape: Dim list from an onnxruntime input or output.

    Returns:
        tuple[int | None, ...]: Integer dims kept; non-int dims mapped to None.
    """
    return tuple(dim if isinstance(dim, int) else None for dim in shape)


def load_onnx_session(
    model_path: str,
    expected_sha256: str | None = None,
    expected_input_shape: tuple[int | None, ...] | None = None,
    expected_output_shape: tuple[int | None, ...] | None = None,
    providers: Sequence[str] | None = None,
) -> OnnxInferenceSession:
    """Open an onnxruntime InferenceSession over model_path.

    Args:
        model_path: Filesystem path to a frozen .onnx artifact.
        expected_sha256: Optional SHA-256 hex digest checked before load.
        expected_input_shape: Optional required input shape after load.
        expected_output_shape: Optional required output shape after load.
        providers: Optional execution-provider list. ``None`` lets onnxruntime
            select from the providers this install registered.

    Returns:
        OnnxInferenceSession: An onnxruntime session matching the protocol.

    Raises:
        ImportError: If onnxruntime is not installed.
        ValueError: If hash or I/O contract verification fails.

    Notes:
        Hash failure rejects the artifact without constructing a session. Shape
        checks run only when both expected shapes are provided.
    """
    if expected_sha256 is not None:
        hash_result = verify_model_hash(model_path, expected_sha256)
        if isinstance(hash_result, Err):
            raise ValueError(f"model hash verification failed ({hash_result.error.value})")
    try:
        import onnxruntime
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is not installed. Install it and provide frozen .onnx "
            "artifacts to use the ONNX backends; use scripted backends in tests "
            "and simulation."
        ) from exc
    kwargs: dict[str, object] = {}
    if providers is not None:
        kwargs["providers"] = list(providers)
    session = cast(OnnxInferenceSession, onnxruntime.InferenceSession(model_path, **kwargs))
    if expected_input_shape is not None and expected_output_shape is not None:
        actual_in = onnx_tensor_shape(session.get_inputs()[0].shape)
        actual_out = onnx_tensor_shape(session.get_outputs()[0].shape)
        contract = verify_io_contract(
            actual_in, actual_out, expected_input_shape, expected_output_shape
        )
        if isinstance(contract, Err):
            raise ValueError(f"model I/O contract verification failed ({contract.error.value})")
    return session
