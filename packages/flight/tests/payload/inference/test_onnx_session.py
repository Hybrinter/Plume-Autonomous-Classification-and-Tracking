"""Tests for the lazy onnxruntime session loader."""

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from flight.payload.inference.onnx_session import load_onnx_session, onnx_tensor_shape


def test_onnx_tensor_shape_maps_symbolic_dims() -> None:
    """Non-integer dims become None; integers are kept."""
    assert onnx_tensor_shape([1, "N", 256]) == (1, None, 256)


@dataclass
class _Node:
    name: str
    shape: list[object]
    type: str = "tensor(float)"


class _Session:
    def __init__(self, inputs: list[_Node], outputs: list[_Node]) -> None:
        self._inputs, self._outputs = inputs, outputs

    def get_inputs(self) -> list[_Node]:
        return self._inputs

    def get_outputs(self) -> list[_Node]:
        return self._outputs

    def run(
        self, output_names: list[str] | None, input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        return []


def _load_stub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, inputs: list[_Node], outputs: list[_Node]
) -> object:
    session = _Session(inputs, outputs)
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(InferenceSession=lambda *_args, **_kwargs: session),
    )
    model = tmp_path / "model.onnx"
    model.write_bytes(b"stub")
    return load_onnx_session(str(model))


def test_loader_accepts_conditioned_dynamic_batch_with_reordered_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Input validation is based on names, independent of ONNX declaration order."""
    result = _load_stub(
        monkeypatch,
        tmp_path,
        [_Node("gsd", ["N", 2]), _Node("image", ["N", 3, 193, 258])],
        [_Node("logits", ["N", 1])],
    )
    assert isinstance(result, _Session)


@pytest.mark.parametrize(
    "inputs",
    [
        [_Node("image", ["N", 3, 193, 258])],
        [_Node("image", ["N", 3, 193, 258]), _Node("scale", ["N", 2])],
        [_Node("image", [1, 3, 193, 258]), _Node("gsd", [1, 2])],
        [_Node("image", ["N", 3, 193.0, 258]), _Node("gsd", ["N", 2])],
        [_Node("image", ["N", 3, 193, 258], "tensor(int64)"), _Node("gsd", ["N", 2])],
    ],
    ids=["single-input", "wrong-name", "fixed-batch", "malformed-dim", "wrong-type"],
)
def test_loader_rejects_stale_or_malformed_conditioned_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    inputs: list[_Node],
) -> None:
    """Legacy one-input and malformed input graphs never load as flight models."""
    outputs = [_Node("logits", ["N", 1])]
    with pytest.raises(ValueError):
        _load_stub(monkeypatch, tmp_path, inputs, outputs)


def test_loader_rejects_wrong_output_rank(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    inputs = [_Node("image", ["N", 3, 193, 258]), _Node("gsd", ["N", 2])]
    with pytest.raises(ValueError):
        _load_stub(monkeypatch, tmp_path, inputs, [_Node("logits", ["N", 1, 193])])


@pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is not None,
    reason="onnxruntime is installed; the absent-runtime guard cannot be exercised",
)
def test_load_onnx_session_requires_onnxruntime_when_absent() -> None:
    """load_onnx_session without onnxruntime raises ImportError."""
    with pytest.raises(ImportError):
        load_onnx_session("model.onnx")
