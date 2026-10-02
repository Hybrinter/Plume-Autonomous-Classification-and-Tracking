"""Tests for two-input ONNX session opening and contract validation.

These tests use a stub session loader so they run without onnxruntime; the
real-SDK parity tests live in test_parity.py and skip when the SDK is absent.
"""

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest
from flight.libs.types import Err, Ok
from tools.ml_models.export.manifest import ModelManifest
from tools.ml_models.export.session import Node, Session, open_session


class _StubSession:
    """In-memory Session stand-in for open_session validation tests."""

    def __init__(self, inputs: list[Node], outputs: list[Node]) -> None:
        self._inputs = inputs
        self._outputs = outputs

    def get_inputs(self) -> Sequence[Node]:
        return self._inputs

    def get_outputs(self) -> Sequence[Node]:
        return self._outputs

    def run(self, output_names: list[str] | None, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        return [feeds["image"]]


def _manifest(**overrides: object) -> ModelManifest:
    fields: dict[str, object] = {
        "version": "a" * 16,
        "kind": "classifier",
        "arch": "pactnet",
        "sha256": "0" * 64,
        "dataset_hash": "b" * 64,
        "band_names": ("BLUE", "GREEN", "RED"),
        "input_shape": (None, 3, None, None),
        "gsd_input_shape": (None, 2),
        "output_shape": (None, 1),
        "gsd_reference_m": 15.87,
        "gsd_min_m": (14.0, 14.0),
        "gsd_max_m": (40.0, 40.0),
        "conditioning": "film-log-gsd-v1",
        "gsd_encoding": "ln_metres_over_reference_lateral_along",
    }
    fields.update(overrides)
    return ModelManifest(**fields)  # type: ignore[arg-type]


def _artifact(tmp_path: Path, manifest: ModelManifest) -> Path:
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"stub-onnx-bytes")
    return artifact


def _manifest_for(path: Path, **overrides: object) -> ModelManifest:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return _manifest(sha256=digest, **overrides)


def _stub_loader(session: Session) -> Callable[[str], Session]:
    return lambda _path: session


def _good_stub() -> _StubSession:
    return _StubSession(
        inputs=[
            Node("image", ("batch", 3, "height", "width"), "tensor(float)"),
            Node("gsd", ("batch", 2), "tensor(float)"),
        ],
        outputs=[Node("logits", ("batch", 1), "tensor(float)")],
    )


def test_open_session_accepts_valid_stub(tmp_path: Path) -> None:
    """A conforming session opens and returns the session object."""
    artifact = _artifact(tmp_path, _manifest())
    manifest = _manifest_for(artifact)
    result = open_session(artifact, manifest, loader=_stub_loader(_good_stub()))
    assert isinstance(result, Ok)
    session = result.value
    outputs = session.run(None, {"image": np.zeros((1, 3, 4, 4), np.float32)})
    assert len(outputs) == 1


def test_open_session_accepts_all_none_batch(tmp_path: Path) -> None:
    """Declared batch dims may all be None instead of a shared symbol."""
    stub = _StubSession(
        inputs=[
            Node("image", (None, 3, "height", "width"), "tensor(float)"),
            Node("gsd", (None, 2), "tensor(float)"),
        ],
        outputs=[Node("logits", (None, 1), "tensor(float)")],
    )
    artifact = _artifact(tmp_path, _manifest())
    result = open_session(artifact, _manifest_for(artifact), loader=_stub_loader(stub))
    assert isinstance(result, Ok)


def test_open_session_rejects_hash_mismatch(tmp_path: Path) -> None:
    """A digest disagreement fails before any session construction."""
    artifact = _artifact(tmp_path, _manifest())
    result = open_session(artifact, _manifest(), loader=_stub_loader(_good_stub()))
    assert isinstance(result, Err)
    assert "sha256" in result.error


def test_open_session_rejects_missing_file(tmp_path: Path) -> None:
    """An unreadable artifact returns Err."""
    result = open_session(tmp_path / "missing.onnx", _manifest(), loader=_stub_loader(_good_stub()))
    assert isinstance(result, Err)


def test_open_session_rejects_fixed_batch(tmp_path: Path) -> None:
    """A fixed integer batch dim is not a dynamic batch."""
    stub = _StubSession(
        inputs=[
            Node("image", (1, 3, 193, 258), "tensor(float)"),
            Node("gsd", (1, 2), "tensor(float)"),
        ],
        outputs=[Node("logits", (1, 1), "tensor(float)")],
    )
    artifact = _artifact(tmp_path, _manifest())
    result = open_session(artifact, _manifest_for(artifact), loader=_stub_loader(stub))
    assert isinstance(result, Err)
    assert "batch" in result.error


def test_open_session_rejects_mixed_batch_symbols(tmp_path: Path) -> None:
    """Different named batch symbols across I/O tensors are rejected."""
    stub = _StubSession(
        inputs=[
            Node("image", ("n_image", 3, "height", "width"), "tensor(float)"),
            Node("gsd", ("n_gsd", 2), "tensor(float)"),
        ],
        outputs=[Node("logits", ("n_image", 1), "tensor(float)")],
    )
    artifact = _artifact(tmp_path, _manifest())
    result = open_session(artifact, _manifest_for(artifact), loader=_stub_loader(stub))
    assert isinstance(result, Err)
    assert "batch" in result.error


@pytest.mark.parametrize(
    ("inputs", "outputs"),
    [
        (
            [Node("image", ("batch", 3, "h", "w"), "tensor(float)")],
            [Node("logits", ("batch", 1), "tensor(float)")],
        ),
        (
            [
                Node("image", ("batch", 3, "h", "w"), "tensor(float)"),
                Node("gsd", ("batch", 2), "tensor(float)"),
                Node("extra", ("batch", 1), "tensor(float)"),
            ],
            [Node("logits", ("batch", 1), "tensor(float)")],
        ),
        (
            [
                Node("image", ("batch", 3, "h", "w"), "tensor(float)"),
                Node("gsd", ("batch", 2), "tensor(float)"),
            ],
            [
                Node("logits", ("batch", 1), "tensor(float)"),
                Node("aux", ("batch", 1), "tensor(float)"),
            ],
        ),
        (
            [
                Node("image", ("batch", 3, "h", "w"), "tensor(int64)"),
                Node("gsd", ("batch", 2), "tensor(float)"),
            ],
            [Node("logits", ("batch", 1), "tensor(float)")],
        ),
    ],
    ids=["missing-gsd", "extra-input", "two-outputs", "wrong-dtype"],
)
def test_open_session_rejects_bad_io_metadata(
    tmp_path: Path, inputs: list[Node], outputs: list[Node]
) -> None:
    """Malformed names, counts, or dtypes are rejected."""
    stub = _StubSession(inputs=inputs, outputs=outputs)
    artifact = _artifact(tmp_path, _manifest())
    result = open_session(artifact, _manifest_for(artifact), loader=_stub_loader(stub))
    assert isinstance(result, Err)


def test_open_session_rejects_shape_mismatch(tmp_path: Path) -> None:
    """Declared shapes differing from the manifest are rejected."""
    stub = _StubSession(
        inputs=[
            Node("image", ("batch", 3, "height", "width"), "tensor(float)"),
            Node("gsd", ("batch", 3), "tensor(float)"),
        ],
        outputs=[Node("logits", ("batch", 1), "tensor(float)")],
    )
    artifact = _artifact(tmp_path, _manifest())
    result = open_session(artifact, _manifest_for(artifact), loader=_stub_loader(stub))
    assert isinstance(result, Err)
    assert "manifest" in result.error


def test_open_session_wraps_loader_errors(tmp_path: Path) -> None:
    """SDK load failures surface as Err, not exceptions."""

    def _boom(_path: str) -> Session:
        raise RuntimeError("simulated ORT failure")

    artifact = _artifact(tmp_path, _manifest())
    result = open_session(artifact, _manifest_for(artifact), loader=_boom)
    assert isinstance(result, Err)
    assert "simulated ORT failure" in result.error


def test_open_session_segmentor(tmp_path: Path) -> None:
    """Segmentor manifests validate a rank-4 logits output."""
    manifest_fields = {
        "kind": "segmentor",
        "arch": "dilatenet_w32_s8",
        "output_shape": (None, 1, None, None),
    }
    stub = _StubSession(
        inputs=[
            Node("image", ("batch", 3, "height", "width"), "tensor(float)"),
            Node("gsd", ("batch", 2), "tensor(float)"),
        ],
        outputs=[Node("logits", ("batch", 1, "height", "width"), "tensor(float)")],
    )
    artifact = _artifact(tmp_path, _manifest())
    result = open_session(
        artifact,
        _manifest_for(artifact, **manifest_fields),
        loader=_stub_loader(stub),
    )
    assert isinstance(result, Ok)
