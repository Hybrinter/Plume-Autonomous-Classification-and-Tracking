"""Torch-to-ONNX export, manifest sidecar, optional INT8 PTQ, and promote.

Graphs emit logits; they do not bake sigmoid.

INT8 export is post-training static quantization with QDQ nodes. Graph input
and output stay float32. onnxruntime loads inside the INT8 path after the
export extra is installed. Calibration tensors convert to numpy only for
``CalibrationDataReader``.

Promote copies an artifact only after `accept_artifact` (or the classifier
gate) reports accepted.

Contains:
  - ExportConfig: frozen export hyperparameters.
  - export: write one ONNX graph plus a Manifest JSON sidecar.
  - write_manifest: serialize a Manifest.
  - int8_artifact_path: sibling ``*.int8.onnx`` path for an FP32 artifact.
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

from tools.inference.accept import Manifest
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
        calib_dir: Processed pack used as INT8 calibration data (train split).
        calib_samples: Maximum calibration tensors.
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
    calib_dir: str = ""
    calib_samples: int = 4


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


def _export_int8(
    fp32_path: Path,
    input_shape: tuple[int, ...],
    base_manifest: Manifest,
    calib_dir: str,
    calib_samples: int,
) -> tuple[Path, Path, Manifest]:
    """Write a sibling INT8 QDQ ONNX file and matching manifest.

    Args:
        fp32_path: FP32 ONNX path produced by ``export``.
        input_shape: Graph input shape ``(1, C, H, W)``.
        base_manifest: FP32 manifest; SHA-256 and quantization are replaced.
        calib_dir: Processed pack directory, or empty for synthetic tensors.
        calib_samples: Maximum calibration tensors.

    Returns:
        tuple: (int8_onnx_path, int8_manifest_path, int8_manifest).

    Raises:
        ImportError: If onnxruntime is not installed.
        ValueError: If calibration tensors do not match the graph input.
    """
    try:
        from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
        from onnxruntime.quantization.calibrate import CalibrationDataReader
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for INT8 export; install pact-tools[export]"
        ) from exc

    batches = _calibration_batches(input_shape, calib_dir, calib_samples)

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

    int8_path = int8_artifact_path(fp32_path)
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=_CalibrationReader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    digest = compute_sha256(str(int8_path))
    int8_manifest = replace(base_manifest, sha256=digest, quantization="int8")
    int8_manifest_path = int8_path.with_suffix(".json")
    write_manifest(str(int8_manifest_path), int8_manifest)
    return int8_path, int8_manifest_path, int8_manifest


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
        is true, a sibling INT8 pair is also written.

    Raises:
        ImportError: If INT8 is requested without onnxruntime.
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
        _export_int8(
            fp32_path=onnx_path,
            input_shape=input_shape,
            base_manifest=manifest,
            calib_dir=config.calib_dir,
            calib_samples=config.calib_samples,
        )
    return onnx_path, manifest_path, manifest


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
