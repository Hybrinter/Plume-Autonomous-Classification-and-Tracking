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
from flight.payload.inference.contract import verify_conditioned_shapes
from flight.payload.inference.verify import verify_model_hash


class OnnxNamedValue(Protocol):
    """Subset of onnxruntime NodeArg used after session load."""

    name: str
    shape: list[object]
    type: str


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
    normalized: list[int | None] = []
    for dim in shape:
        if dim is None or isinstance(dim, str):
            normalized.append(None)
        elif type(dim) is int and dim > 0:
            normalized.append(dim)
        else:
            raise ValueError(f"invalid ONNX dimension: {dim!r}")
    return tuple(normalized)


def load_onnx_session(
    model_path: str,
    expected_sha256: str | None = None,
    expected_input_shape: tuple[int | None, ...] | None = None,
    expected_output_shape: tuple[int | None, ...] | None = None,
    providers: Sequence[str] | None = None,
    expected_gsd_shape: tuple[int | None, ...] | None = None,
) -> OnnxInferenceSession:
    """Open an onnxruntime InferenceSession over model_path.

    Args:
        model_path: Filesystem path to a frozen .onnx artifact.
        expected_sha256: Optional SHA-256 hex digest checked before load.
        expected_input_shape: Optional required input shape after load.
        expected_output_shape: Optional required output shape after load.
        providers: Optional execution-provider list. ``None`` lets onnxruntime
            select from the providers this install registered.
        expected_gsd_shape: Optional required GSD input shape; defaults to
            the conditioned flight contract ``(None, 2)``.

    Returns:
        OnnxInferenceSession: An onnxruntime session matching the protocol.

    Raises:
        ImportError: If onnxruntime is not installed.
        ValueError: If hash or I/O contract verification fails.

    Notes:
        Hash failure rejects the artifact without constructing a session. Shape
        every loaded graph is checked against the conditioned flight contract.
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
    inputs = list(session.get_inputs())
    outputs = list(session.get_outputs())
    names = [node.name for node in inputs]
    if len(inputs) != 2 or len(set(names)) != 2 or set(names) != {"image", "gsd"}:
        raise ValueError("model I/O contract requires exactly image and gsd inputs")
    if len(outputs) != 1:
        raise ValueError("model I/O contract requires exactly one output")
    if any(node.type != "tensor(float)" for node in (*inputs, *outputs)):
        raise ValueError("model I/O contract requires float32 image, gsd, and output tensors")
    by_name = {node.name: node for node in inputs}
    actual_image = onnx_tensor_shape(by_name["image"].shape)
    actual_gsd = onnx_tensor_shape(by_name["gsd"].shape)
    actual_output = onnx_tensor_shape(outputs[0].shape)
    channels = actual_image[1] if len(actual_image) == 4 else 0
    kind = "classifier" if len(actual_output) == 2 else "segmentor"
    tile_hw = None
    if expected_input_shape is not None:
        if len(expected_input_shape) != 4:
            raise ValueError("expected_input_shape must be rank four")
        if expected_input_shape[2] is not None and expected_input_shape[3] is not None:
            tile_hw = (expected_input_shape[2], expected_input_shape[3])
    contract = verify_conditioned_shapes(
        actual_image, actual_gsd, actual_output, channels or 0, tile_hw, kind
    )
    if isinstance(contract, Err):
        raise ValueError(f"model I/O contract verification failed ({contract.error.value})")
    if expected_input_shape is not None and actual_image != expected_input_shape:
        raise ValueError("model image input shape does not match expected_input_shape")
    if expected_gsd_shape is not None and actual_gsd != expected_gsd_shape:
        raise ValueError("model GSD input shape does not match expected_gsd_shape")
    if expected_output_shape is not None and actual_output != expected_output_shape:
        raise ValueError("model output shape does not match expected_output_shape")
    return session
