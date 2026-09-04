"""Torch-to-ONNX export, manifest sidecar, FP16/INT8 conversion, and promote.

Graphs emit logits; they do not bake sigmoid.

INT8 export is post-training static quantization with QDQ nodes. FP16 conversion
rewrites weights and activations while keeping graph input and output float32.
onnxruntime loads inside those paths after the export extra is installed.
Calibration tensors convert to numpy only for ``CalibrationDataReader``.

The quality knee is classifier FP16 plus segmentor INT8. ``quantize_knee``
writes that pair over existing factory artifacts.

Promote copies an artifact only after `accept_artifact` (or the classifier
gate) reports accepted.

Contains:
  - ExportConfig: frozen export hyperparameters.
  - export: write one ONNX graph plus a Manifest JSON sidecar.
  - reexport_spatial: rebuild a graph at a new H/W and copy matching weights.
  - convert_fp16: rewrite an FP32 graph to FP16 with float32 I/O.
  - quantize_int8: static QDQ INT8 with float32 I/O.
  - quantize_knee: classifier FP16 and segmentor INT8 in place.
  - write_manifest: serialize a Manifest.
  - int8_artifact_path: sibling ``*.int8.onnx`` path for an FP32 artifact.
  - fp16_artifact_path: sibling ``*.fp16.onnx`` path for an FP32 artifact.
  - promote: copy a passed artifact into the destination path.
  - GateReport: protocol with accepted + detail.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from flight.payload.inference.verify import compute_sha256
from torch import Tensor, nn

from tools.inference.accept import Manifest, load_manifest
from tools.inference.arch.registry import build
from tools.inference.data import _row_image, load_processed_pack

_EXPORT_KINDS = frozenset({"classifier", "segmentor"})


class GateReport(Protocol):
    """Minimum acceptance report fields required to promote an artifact."""

    @property
    def accepted(self) -> bool:
        """Whether the gate passed."""
        ...

    @property
    def detail(self) -> str:
        """Human-readable per-check summary."""
        ...


@dataclass(frozen=True, slots=True)
class ExportConfig:
    """Frozen export hyperparameters.

    Attributes:
        kind: ``classifier`` or ``segmentor``.
        checkpoint_path: Trained ``.pt`` path.
        output_path: Destination FP32 ``.onnx`` path.
        in_channels: Fallback channel count when the checkpoint omits it.
        input_height_px: Fallback height when the checkpoint omits it.
        input_width_px: Fallback width when the checkpoint omits it.
        version: Manifest version string.
        model_repo_sha: Source revision recorded in the manifest.
        dataset_hash: Training-pack digest recorded in the manifest.
        opset: ONNX opset for ``torch.onnx.export``.
        int8: When true, also write a sibling INT8 QDQ artifact.
        fp16: When true, also write a sibling FP16 artifact.
        calib_dir: Processed pack used as INT8 calibration data (train split).
        calib_samples: Maximum calibration tensors.
        override_spatial: When true, use ``input_height_px`` / ``input_width_px``
            even if the checkpoint recorded a different size.
    """

    kind: str
    checkpoint_path: str
    output_path: str
    in_channels: int = 4
    input_height_px: int = 256
    input_width_px: int = 256
    version: str = "v1"
    model_repo_sha: str = "unknown"
    dataset_hash: str = "synthetic"
    opset: int = 17
    int8: bool = False
    fp16: bool = False
    calib_dir: str = ""
    calib_samples: int = 4
    override_spatial: bool = False


def write_manifest(path: str, manifest: Manifest) -> None:
    """Write a Manifest as JSON.

    Args:
        path: Destination JSON path.
        manifest: Parsed manifest fields.

    Returns:
        None.
    """
    payload = {
        "version": manifest.version,
        "model_repo_sha": manifest.model_repo_sha,
        "dataset_hash": manifest.dataset_hash,
        "input_shape": list(manifest.input_shape),
        "output_shape": list(manifest.output_shape),
        "sha256": manifest.sha256,
        "quantization": manifest.quantization,
    }
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def int8_artifact_path(fp32_path: str | Path) -> Path:
    """Return the sibling INT8 ONNX path for an FP32 artifact.

    Args:
        fp32_path: FP32 ``.onnx`` path.

    Returns:
        Path: ``<stem>.int8.onnx`` next to the FP32 file.
    """
    path = Path(fp32_path)
    return path.with_name(f"{path.stem}.int8{path.suffix}")


def fp16_artifact_path(fp32_path: str | Path) -> Path:
    """Return the sibling FP16 ONNX path for an FP32 artifact.

    Args:
        fp32_path: FP32 ``.onnx`` path.

    Returns:
        Path: ``<stem>.fp16.onnx`` next to the FP32 file.
    """
    path = Path(fp32_path)
    return path.with_name(f"{path.stem}.fp16{path.suffix}")


def _staging_path(source: Path, dest: Path) -> Path:
    """Return ``dest``, or a sibling temp path when dest would overwrite source."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.resolve() == source.resolve():
        return dest.with_name(f".{dest.name}.quant")
    return dest


def _commit_staging(staging: Path, dest: Path) -> None:
    """Move ``staging`` onto ``dest`` when they are different paths."""
    if staging.resolve() != dest.resolve():
        staging.replace(dest)


def _manifest_after_quant(
    source: Path,
    dest: Path,
    quantization: str,
) -> Manifest:
    """Copy provenance from the source sidecar and set digest plus quantization.

    Args:
        source: Source ONNX path (sidecar must exist).
        dest: Written ONNX path to hash.
        quantization: Manifest ``quantization`` field.

    Returns:
        Manifest for ``dest``.

    Raises:
        FileNotFoundError: If the source sidecar is missing.
    """
    sidecar = source.with_suffix(".json")
    if not sidecar.is_file():
        raise FileNotFoundError(str(sidecar))
    base = load_manifest(str(sidecar))
    return replace(base, sha256=compute_sha256(str(dest)), quantization=quantization)


def _calibration_batches(
    input_shape: tuple[int, ...],
    calib_dir: str,
    calib_samples: int,
) -> list[torch.Tensor]:
    """Return NCHW float32 calibration tensors in ``[0, 1]``.

    Args:
        input_shape: Model input shape ``(1, C, H, W)``.
        calib_dir: Processed pack directory, or empty for synthetic random data.
        calib_samples: Maximum tensor count.

    Returns:
        list[torch.Tensor]: Each item is ``(1, C, H, W)`` float32 on CPU.

    Raises:
        ValueError: If a pack image geometry does not match ``input_shape``.
        FileNotFoundError: If ``calib_dir`` is set and the pack is missing.
    """
    channels = int(input_shape[1])
    height = int(input_shape[2])
    width = int(input_shape[3])
    n = max(int(calib_samples), 1)
    expected = (channels, height, width)
    if calib_dir:
        pack = load_processed_pack(calib_dir, load_masks=False)
        indices = pack.splits.train[:n]
        if not indices:
            raise ValueError("INT8 calibration pack has an empty train split")
        batches: list[torch.Tensor] = []
        for idx in indices:
            image = _row_image(pack.images, idx)
            if tuple(int(dim) for dim in image.shape) != expected:
                raise ValueError(f"calibration image shape {tuple(image.shape)} != {expected}")
            batches.append(image.unsqueeze(0).to(dtype=torch.float32))
        return batches
    generator = torch.Generator()
    generator.manual_seed(0)
    return [
        torch.rand((1, channels, height, width), generator=generator, dtype=torch.float32)
        for _ in range(n)
    ]


def quantize_int8(
    source_onnx: str,
    dest_onnx: str,
    *,
    calib_dir: str = "",
    calib_samples: int = 4,
) -> tuple[Path, Path, Manifest]:
    """Write a QDQ INT8 ONNX file and matching manifest. I/O stay float32.

    Args:
        source_onnx: FP32 ONNX path.
        dest_onnx: Destination INT8 path. May equal ``source_onnx``.
        calib_dir: Processed pack directory, or empty for synthetic tensors.
        calib_samples: Maximum calibration tensors.

    Returns:
        tuple: (int8_onnx_path, int8_manifest_path, int8_manifest).

    Raises:
        ImportError: If onnxruntime is not installed.
        FileNotFoundError: If the source ONNX or sidecar is missing.
        ValueError: If calibration tensors do not match the graph input.
    """
    try:
        from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
        from onnxruntime.quantization.calibrate import CalibrationDataReader
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for INT8 export; install pact-tools[export]"
        ) from exc

    source = Path(source_onnx)
    if not source.is_file():
        raise FileNotFoundError(source_onnx)
    dest = Path(dest_onnx)
    sidecar = source.with_suffix(".json")
    if not sidecar.is_file():
        raise FileNotFoundError(str(sidecar))
    base = load_manifest(str(sidecar))
    if any(dim is None for dim in base.input_shape):
        raise ValueError(f"INT8 calibration needs a concrete input shape, got {base.input_shape}")
    concrete = tuple(int(dim) for dim in base.input_shape if dim is not None)
    batches = _calibration_batches(concrete, calib_dir, calib_samples)

    class _CalibrationReader(CalibrationDataReader):  # type: ignore[misc]
        """Yields ``{input: tensor}`` dicts, then None."""

        def __init__(self) -> None:
            self._index = 0

        def get_next(self) -> dict[str, np.ndarray] | None:
            if self._index >= len(batches):
                return None
            array = np.ascontiguousarray(
                batches[self._index].detach().cpu().numpy(), dtype=np.float32
            )
            item = {"input": array}
            self._index += 1
            return item

        def rewind(self) -> None:
            self._index = 0

    staging = _staging_path(source, dest)
    quantize_static(
        model_input=str(source),
        model_output=str(staging),
        calibration_data_reader=_CalibrationReader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    _commit_staging(staging, dest)
    manifest = _manifest_after_quant(source, dest, "int8")
    manifest_path = dest.with_suffix(".json")
    write_manifest(str(manifest_path), manifest)
    return dest, manifest_path, manifest


def convert_fp16(source_onnx: str, dest_onnx: str) -> tuple[Path, Path, Manifest]:
    """Rewrite an FP32 ONNX graph to FP16. Graph I/O stay float32.

    Args:
        source_onnx: FP32 ONNX path.
        dest_onnx: Destination FP16 path. May equal ``source_onnx``.

    Returns:
        tuple: (fp16_onnx_path, fp16_manifest_path, fp16_manifest).

    Raises:
        ImportError: If onnx or onnxruntime is not installed.
        FileNotFoundError: If the source ONNX or sidecar is missing.
    """
    try:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16
    except ImportError as exc:
        raise ImportError(
            "onnx and onnxruntime are required for FP16 conversion; install pact-tools[export]"
        ) from exc

    source = Path(source_onnx)
    if not source.is_file():
        raise FileNotFoundError(source_onnx)
    dest = Path(dest_onnx)
    sidecar = source.with_suffix(".json")
    if not sidecar.is_file():
        raise FileNotFoundError(str(sidecar))
    staging = _staging_path(source, dest)
    model = onnx.load(str(source))
    model_fp16 = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(staging))
    _commit_staging(staging, dest)
    manifest = _manifest_after_quant(source, dest, "fp16")
    manifest_path = dest.with_suffix(".json")
    write_manifest(str(manifest_path), manifest)
    return dest, manifest_path, manifest


def quantize_knee(
    classifier_onnx: str,
    segmentor_onnx: str,
    *,
    calib_dir: str = "",
    calib_samples: int = 4,
) -> tuple[tuple[Path, Path, Manifest], tuple[Path, Path, Manifest]]:
    """Write the quality-knee pair: classifier FP16, segmentor INT8.

    Both destinations default to overwriting the source factory paths.
    Graph input and output stay float32.

    Args:
        classifier_onnx: Classifier ONNX path, overwritten with FP16.
        segmentor_onnx: Segmentor ONNX path, overwritten with INT8 QDQ.
        calib_dir: INT8 calibration pack, or empty for synthetic tensors.
        calib_samples: Maximum INT8 calibration tensors.

    Returns:
        tuple: ``((cls_onnx, cls_json, cls_manifest), (seg_onnx, seg_json,
        seg_manifest))``.
    """
    classifier = convert_fp16(classifier_onnx, classifier_onnx)
    segmentor = quantize_int8(
        segmentor_onnx,
        segmentor_onnx,
        calib_dir=calib_dir,
        calib_samples=calib_samples,
    )
    return classifier, segmentor


def _build_model(kind: str, arch: str, in_channels: int) -> nn.Module:
    """Construct the network for `kind` and `arch`."""
    return build(kind, arch, in_channels)


def _output_shape(kind: str, height: int, width: int) -> tuple[int, ...]:
    """Return the frozen output shape for a kind."""
    if kind == "classifier":
        return (1, 1)
    if kind == "segmentor":
        return (1, 1, height, width)
    raise ValueError(f"unknown export kind {kind!r}")


def _onnx_export(model: nn.Module, dummy: Tensor, output_path: str, opset: int) -> None:
    """Write an ONNX graph with logits named ``logits``."""
    try:
        torch.onnx.export(
            model,
            (dummy,),
            output_path,
            input_names=["input"],
            output_names=["logits"],
            opset_version=opset,
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            model,
            (dummy,),
            output_path,
            input_names=["input"],
            output_names=["logits"],
            opset_version=opset,
        )


def export(config: ExportConfig) -> tuple[Path, Path, Manifest]:
    """Export a checkpoint to ONNX logits and write a matching manifest.

    Args:
        config: Export hyperparameters.

    Returns:
        tuple: FP32 ``(onnx_path, manifest_path, manifest)``. When ``config.int8``
        is true, a sibling INT8 pair is also written. When ``config.fp16`` is
        true, a sibling FP16 pair is also written.

    Raises:
        ImportError: If INT8 or FP16 is requested without onnxruntime.
        ValueError: If `config.kind` is unknown.
        FileNotFoundError: If the checkpoint is missing.
    """
    if config.kind not in _EXPORT_KINDS:
        raise ValueError(f"unknown export kind {config.kind!r}")
    try:
        payload = torch.load(config.checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(config.checkpoint_path, map_location="cpu")
    in_channels = int(payload.get("in_channels", config.in_channels))
    if config.override_spatial:
        height = int(config.input_height_px)
        width = int(config.input_width_px)
    else:
        height = int(payload.get("input_height_px", config.input_height_px))
        width = int(payload.get("input_width_px", config.input_width_px))
    kind = str(payload.get("kind", config.kind))
    if kind not in _EXPORT_KINDS:
        raise ValueError(f"unknown checkpoint kind {kind!r}")
    arch = str(payload.get("arch", ""))
    model = _build_model(kind, arch, in_channels)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    dummy = torch.zeros(1, in_channels, height, width, dtype=torch.float32)
    onnx_path = Path(config.output_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        _onnx_export(model, dummy, str(onnx_path), config.opset)
    digest = compute_sha256(str(onnx_path))
    input_shape = (1, in_channels, height, width)
    output_shape = _output_shape(kind, height, width)
    manifest = Manifest(
        version=config.version,
        model_repo_sha=config.model_repo_sha,
        dataset_hash=config.dataset_hash,
        input_shape=input_shape,
        output_shape=output_shape,
        sha256=digest,
        quantization="fp32",
    )
    manifest_path = onnx_path.with_suffix(".json")
    write_manifest(str(manifest_path), manifest)
    if config.int8:
        quantize_int8(
            str(onnx_path),
            str(int8_artifact_path(onnx_path)),
            calib_dir=config.calib_dir,
            calib_samples=config.calib_samples,
        )
    if config.fp16:
        convert_fp16(str(onnx_path), str(fp16_artifact_path(onnx_path)))
    return onnx_path, manifest_path, manifest


def _copy_matching_initializers(source_onnx: str, dest_onnx: str) -> int:
    """Copy same-name, same-shape initializers from ``source_onnx`` into ``dest_onnx``.

    Args:
        source_onnx: Trained ONNX path (any spatial size).
        dest_onnx: Freshly exported graph whose weights are replaced in place.

    Returns:
        int: Number of tensors copied.

    Raises:
        ImportError: If onnx is not installed.
        ValueError: If no matching initializer can be copied.
    """
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:
        raise ImportError(
            "onnx is required for spatial re-export; install pact-tools[export]"
        ) from exc

    src = onnx.load(source_onnx)
    dst = onnx.load(dest_onnx)
    src_init = {init.name: init for init in src.graph.initializer}
    copied = 0
    for init in dst.graph.initializer:
        src_tensor = src_init.get(init.name)
        if src_tensor is None:
            continue
        src_arr = numpy_helper.to_array(src_tensor)
        dst_arr = numpy_helper.to_array(init)
        if src_arr.shape != dst_arr.shape:
            continue
        init.CopyFrom(numpy_helper.from_array(src_arr, name=init.name))
        copied += 1
    if copied == 0:
        raise ValueError(f"no matching ONNX initializers to copy from {source_onnx}")
    onnx.save(dst, dest_onnx)
    return copied


def reexport_spatial(
    source_onnx: str,
    dest_onnx: str,
    *,
    kind: str,
    arch: str,
    height: int,
    width: int,
    in_channels: int = 4,
    opset: int = 17,
    version: str | None = None,
    model_repo_sha: str | None = None,
    dataset_hash: str | None = None,
) -> tuple[Path, Path, Manifest]:
    """Export ``arch`` at ``(H, W)`` and copy matching weights from ``source_onnx``.

    Convolution weights do not include spatial size, so a fully convolutional
    graph can change H/W without retraining. Provenance fields come from the
    source sidecar when present.

    Args:
        source_onnx: Existing FP32 ONNX path whose weights are copied.
        dest_onnx: Destination ONNX path (may equal ``source_onnx``).
        kind: ``classifier`` or ``segmentor``.
        arch: Architecture name passed to ``build``.
        height: New input height in pixels.
        width: New input width in pixels.
        in_channels: Input channel count.
        opset: ONNX opset for ``torch.onnx.export``.
        version: Manifest version; defaults to the source sidecar.
        model_repo_sha: Manifest revision; defaults to the source sidecar.
        dataset_hash: Manifest dataset digest; defaults to the source sidecar.

    Returns:
        tuple: ``(onnx_path, manifest_path, manifest)`` at the new spatial size.

    Raises:
        ValueError: If ``kind`` is unknown or no weights copy.
        FileNotFoundError: If ``source_onnx`` is missing.
        ImportError: If onnx is not installed.
    """
    if kind not in _EXPORT_KINDS:
        raise ValueError(f"unknown export kind {kind!r}")
    source = Path(source_onnx)
    if not source.is_file():
        raise FileNotFoundError(source_onnx)
    src_manifest: Manifest | None = None
    sidecar = source.with_suffix(".json")
    if sidecar.is_file():
        src_manifest = load_manifest(str(sidecar))
    dest = Path(dest_onnx)
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest
    if dest.resolve() == source.resolve():
        staging = dest.with_name(f".{dest.name}.reexport")
    model = _build_model(kind, arch, in_channels)
    model.eval()
    dummy = torch.zeros(1, in_channels, height, width, dtype=torch.float32)
    with torch.no_grad():
        _onnx_export(model, dummy, str(staging), opset)
    _copy_matching_initializers(str(source), str(staging))
    if staging != dest:
        staging.replace(dest)
    digest = compute_sha256(str(dest))
    input_shape = (1, in_channels, height, width)
    output_shape = _output_shape(kind, height, width)
    manifest = Manifest(
        version=version
        if version is not None
        else (src_manifest.version if src_manifest else "v1"),
        model_repo_sha=(
            model_repo_sha
            if model_repo_sha is not None
            else (src_manifest.model_repo_sha if src_manifest else "unknown")
        ),
        dataset_hash=(
            dataset_hash
            if dataset_hash is not None
            else (src_manifest.dataset_hash if src_manifest else "synthetic")
        ),
        input_shape=input_shape,
        output_shape=output_shape,
        sha256=digest,
        quantization="fp32",
    )
    manifest_path = dest.with_suffix(".json")
    write_manifest(str(manifest_path), manifest)
    return dest, manifest_path, manifest


def promote(artifact_path: str, dest_path: str, report: GateReport) -> Path:
    """Copy a frozen artifact into `dest_path` only when the gate passed.

    Args:
        artifact_path: Source .onnx path.
        dest_path: Destination path (for example data/models/active_segmentor.onnx).
        report: Gate result. `accepted` must be True.

    Returns:
        Path: The destination path.

    Raises:
        ValueError: If the report is not accepted.
    """
    if not report.accepted:
        raise ValueError(f"refusing to promote a rejected artifact: {report.detail}")
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact_path, dest)
    sidecar = Path(artifact_path).with_suffix(".json")
    if sidecar.is_file():
        shutil.copy2(sidecar, dest.with_suffix(".json"))
    return dest
