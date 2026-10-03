"""Tests for the FP16 and INT8 precision conversions."""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from flight.libs.types import Err, Ok
from tools.ml_models.export.contract import CONDITIONING_ID, GSD_ENCODING
from tools.ml_models.export.manifest import ModelManifest, write_manifest
from tools.ml_models.export.precision import convert_fp16, quantize_int8
from tools.ml_models.export.session import Node, Session


class _StubSession:
    """Minimal Session stand-in for precision tests."""

    def get_inputs(self) -> Sequence[Node]:
        return [
            Node("image", ("batch", 3, "height", "width"), "tensor(float)"),
            Node("gsd", ("batch", 2), "tensor(float)"),
        ]

    def get_outputs(self) -> Sequence[Node]:
        return [Node("logits", ("batch", 1), "tensor(float)")]

    def run(self, output_names: list[str] | None, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        return [feeds["image"]]


def _manifest(**overrides: object) -> ModelManifest:
    fields: dict[str, object] = {
        "version": "a" * 16,
        "kind": "classifier",
        "arch": "pactnet_stub",
        "sha256": "0" * 64,
        "dataset_hash": "b" * 64,
        "band_names": ("BLUE", "GREEN", "RED"),
        "input_shape": (None, 3, None, None),
        "gsd_input_shape": (None, 2),
        "output_shape": (None, 1),
        "gsd_reference_m": 15.87,
        "gsd_min_m": (14.0, 14.0),
        "gsd_max_m": (40.0, 40.0),
        "conditioning": CONDITIONING_ID,
        "gsd_encoding": GSD_ENCODING,
    }
    fields.update(overrides)
    return ModelManifest(**fields)  # type: ignore[arg-type]


def _source(tmp_path: Path, payload: bytes = b"fp32-graph") -> Path:
    """Write a source artifact plus a sidecar hashed to its bytes."""
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(payload)
    manifest = _manifest(sha256=hashlib.sha256(payload).hexdigest())
    write_manifest(tmp_path / "model.json", manifest)
    return artifact


def _stub_open_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace open_session with a stub that returns Ok on real bytes."""
    import tools.ml_models.export.precision as precision

    def fake(artifact: Path, manifest: ModelManifest) -> Ok[Session]:
        return Ok(_StubSession())

    monkeypatch.setattr(precision, "open_session", fake)


def test_fp16_converts_and_rewrites_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A converted artifact gets a fresh hash, version, and quantization."""
    import tools.ml_models.export.precision as precision

    source = _source(tmp_path)
    dest = tmp_path / "model-fp16.onnx"
    _stub_open_session(monkeypatch)

    def fake_write(src: Path, tmp: Path) -> None:
        assert src == source
        tmp.write_bytes(b"fp16-graph")

    monkeypatch.setattr(precision, "_write_fp16", fake_write)
    result = convert_fp16(source, dest)
    assert isinstance(result, Ok)
    assert result.value == dest
    assert dest.read_bytes() == b"fp16-graph"
    sidecar = json.loads((tmp_path / "model-fp16.json").read_text(encoding="utf-8"))
    expected_sha = hashlib.sha256(b"fp16-graph").hexdigest()
    assert sidecar["sha256"] == expected_sha
    assert sidecar["version"] == expected_sha[:16]
    assert sidecar["quantization"] == "fp16"
    assert sidecar["kind"] == "classifier"
    assert sidecar["conditioning"] == CONDITIONING_ID
    assert sidecar["dataset_hash"] == "b" * 64
    # Source artifact and sidecar are untouched.
    assert source.read_bytes() == b"fp32-graph"
    original = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
    assert original["quantization"] == "fp32"


def test_int8_converts_with_calibration_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INT8 passes the collected calibration batches to the writer."""
    import tools.ml_models.export.precision as precision

    source = _source(tmp_path)
    dest = tmp_path / "model-int8.onnx"
    _stub_open_session(monkeypatch)
    batches = [
        {
            "image": np.zeros((1, 3, 8, 8), dtype=np.float32),
            "gsd": np.zeros((1, 2), dtype=np.float32),
        }
        for _ in range(4)
    ]
    seen: dict[str, object] = {}

    def fake_calibration(
        dataset: str | Path, model: ModelManifest, samples: int
    ) -> list[dict[str, np.ndarray]]:
        seen["dataset"] = dataset
        seen["samples"] = samples
        return batches

    def fake_write(src: Path, tmp: Path, fed: list[dict[str, np.ndarray]]) -> None:
        seen["batches"] = fed
        tmp.write_bytes(b"int8-graph")

    monkeypatch.setattr(precision, "calibration_batches", fake_calibration)
    monkeypatch.setattr(precision, "_write_int8", fake_write)
    dataset = tmp_path / "ds"
    result = quantize_int8(source, dest, dataset=dataset, calib_samples=4)
    assert isinstance(result, Ok)
    assert seen["dataset"] == dataset
    assert seen["samples"] == 4
    assert seen["batches"] is batches
    assert dest.read_bytes() == b"int8-graph"
    sidecar = json.loads((tmp_path / "model-int8.json").read_text(encoding="utf-8"))
    assert sidecar["quantization"] == "int8"
    assert sidecar["sha256"] == hashlib.sha256(b"int8-graph").hexdigest()


def test_destinations_must_be_new_and_distinct(tmp_path: Path) -> None:
    """Same-path and existing destinations are refused; nothing is written."""
    source = _source(tmp_path)
    assert isinstance(convert_fp16(source, source), Err)
    existing = tmp_path / "taken.onnx"
    existing.write_bytes(b"occupied")
    result = convert_fp16(source, existing)
    assert isinstance(result, Err)
    assert existing.read_bytes() == b"occupied"
    sidecar_taken = tmp_path / "taken2.onnx"
    (tmp_path / "taken2.json").write_text("{}", encoding="utf-8")
    assert isinstance(convert_fp16(source, sidecar_taken), Err)
    assert not sidecar_taken.exists()


def test_missing_source_and_failed_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing inputs and a source that fails open_session return Err."""
    import tools.ml_models.export.precision as precision

    missing = tmp_path / "missing.onnx"
    assert isinstance(convert_fp16(missing, tmp_path / "out.onnx"), Err)
    source = _source(tmp_path)

    def failing(artifact: Path, manifest: ModelManifest) -> Err[str]:
        return Err("graph invalid")

    monkeypatch.setattr(precision, "open_session", failing)
    dest = tmp_path / "out.onnx"
    result = convert_fp16(source, dest)
    assert isinstance(result, Err)
    assert "graph invalid" in result.error
    assert not dest.exists()
    assert not dest.with_suffix(".json").exists()
    assert not list(tmp_path.glob("*.partial"))


def test_int8_empty_calibration_and_missing_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing dataset and an empty batch set return Err."""
    import tools.ml_models.export.precision as precision

    source = _source(tmp_path)
    assert isinstance(quantize_int8(source, tmp_path / "x.onnx", dataset=tmp_path / "missing"), Err)

    monkeypatch.setattr(precision, "calibration_batches", lambda *args: [])
    result = quantize_int8(source, tmp_path / "y.onnx", dataset=tmp_path / "ds")
    assert isinstance(result, Err)
    assert "no batches" in result.error
    assert not (tmp_path / "y.onnx").exists()


def test_manifest_rejects_unknown_quantization() -> None:
    """The closed quantization set is fp32, fp16, and int8 only."""
    with pytest.raises(ValueError, match="quantization"):
        _manifest(quantization="int4")


def _real_graph(tmp_path: Path) -> Path:
    """Write a small two-input FP32 Conv graph and its sidecar.

    Returns:
        Path: The artifact path; requires onnx and onnxruntime.
    """
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    weight = onnx.numpy_helper.from_array(np.full((1, 3, 1, 1), 0.5, dtype=np.float32), "w")
    node = onnx.helper.make_node("Conv", ["image", "w"], ["logits"])
    graph = onnx.helper.make_graph(
        [node],
        "g",
        [
            onnx.helper.make_tensor_value_info(
                "image", onnx.TensorProto.FLOAT, ["batch", 3, 193, 258]
            ),
            onnx.helper.make_tensor_value_info("gsd", onnx.TensorProto.FLOAT, ["batch", 2]),
        ],
        [
            onnx.helper.make_tensor_value_info(
                "logits", onnx.TensorProto.FLOAT, ["batch", 1, 193, 258]
            )
        ],
        [weight],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    artifact = tmp_path / "fp32.onnx"
    onnx.save(model, str(artifact))
    manifest = _manifest(
        kind="segmentor",
        arch="dilatenet_stub",
        input_shape=(None, 3, 193, 258),
        output_shape=(None, 1, 193, 258),
        sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
    )
    write_manifest(tmp_path / "fp32.json", manifest)
    return artifact


def test_fp16_real_sdk_converts_weights_keeps_io(tmp_path: Path) -> None:
    """Real ONNX: weights go float16 while graph I/O stays float32."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    source = _real_graph(tmp_path)
    dest = tmp_path / "fp16.onnx"
    result = convert_fp16(source, dest)
    assert isinstance(result, Ok)
    model = onnx.load(str(dest))
    assert model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    weights = {init.name: init for init in model.graph.initializer}
    assert weights["w"].data_type == onnx.TensorProto.FLOAT16
    sidecar = json.loads((tmp_path / "fp16.json").read_text(encoding="utf-8"))
    assert sidecar["quantization"] == "fp16"


def test_int8_real_sdk_quantizes_over_two_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real QDQ quantization over stubbed image+gsd calibration pairs."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import tools.ml_models.export.precision as precision

    source = _real_graph(tmp_path)
    dest = tmp_path / "int8.onnx"
    rng = np.random.default_rng(0)
    batches = [
        {
            "image": rng.random((1, 3, 193, 258), dtype=np.float32),
            "gsd": np.zeros((1, 2), dtype=np.float32),
        }
        for _ in range(2)
    ]
    fed: list[dict[str, np.ndarray]] = []

    real_write = precision._write_int8

    def spy_write(src: Path, tmp: Path, got: list[dict[str, np.ndarray]]) -> None:
        fed.extend(got)
        real_write(src, tmp, got)

    monkeypatch.setattr(precision, "calibration_batches", lambda *args: batches)
    monkeypatch.setattr(precision, "_write_int8", spy_write)
    result = quantize_int8(source, dest, dataset=tmp_path / "ds", calib_samples=2)
    assert isinstance(result, Ok)
    assert [set(batch) for batch in fed] == [{"image", "gsd"}] * 2
    sidecar = json.loads((tmp_path / "int8.json").read_text(encoding="utf-8"))
    assert sidecar["quantization"] == "int8"
