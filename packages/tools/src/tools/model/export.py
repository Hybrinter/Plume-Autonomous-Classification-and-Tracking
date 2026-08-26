"""Torch-to-ONNX export, manifest sidecar, and promote into data/models/.

Importing this module does not import torch. Torch loads inside `export` after
the train extra is installed. Graphs emit logits; they do not bake sigmoid.

Promote copies an artifact only after `accept_artifact` (or the classifier
gate) reports accepted.

Contains:
  - ExportConfig: frozen export hyperparameters.
  - export: write one ONNX graph plus a Manifest JSON sidecar.
  - write_manifest: serialize a Manifest.
  - promote: copy a passed artifact into the destination path.
  - GateReport: protocol with accepted + detail.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Protocol

from flight.payload.inference.verify import compute_sha256

from tools.model.accept import Manifest

if TYPE_CHECKING:
    from torch import Tensor, nn

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
    """Frozen export hyperparameters."""

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
    }
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _import_torch() -> ModuleType:
    """Import torch or raise a tools-extra error."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required for tools.model.export; install pact-tools[train]"
        ) from exc
    return torch


def _build_model(kind: str, in_channels: int) -> nn.Module:
    """Construct the network for `kind` (lazy arch import)."""
    if kind == "classifier":
        from tools.model.arch.classifier import build_classifier

        return build_classifier(in_channels=in_channels)
    if kind == "segmentor":
        from tools.model.arch.unet import build_segmentor

        return build_segmentor(in_channels=in_channels, out_channels=1)
    raise ValueError(f"unknown export kind {kind!r}")


def _output_shape(kind: str, height: int, width: int) -> tuple[int, ...]:
    """Return the frozen output shape for a kind."""
    if kind == "classifier":
        return (1, 1)
    if kind == "segmentor":
        return (1, 1, height, width)
    raise ValueError(f"unknown export kind {kind!r}")


def _onnx_export(model: nn.Module, dummy: Tensor, output_path: str, opset: int) -> None:
    """Write an ONNX graph with logits named ``logits``."""
    torch = _import_torch()
    try:
        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=["input"],
            output_names=["logits"],
            opset_version=opset,
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            model,
            dummy,
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
        tuple: (onnx_path, manifest_path, manifest).

    Raises:
        ImportError: If torch is not installed.
        ValueError: If `config.kind` is unknown.
        FileNotFoundError: If the checkpoint is missing.
    """
    if config.kind not in _EXPORT_KINDS:
        raise ValueError(f"unknown export kind {config.kind!r}")
    torch = _import_torch()
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
    model = _build_model(kind, in_channels)
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
    )
    manifest_path = onnx_path.with_suffix(".json")
    write_manifest(str(manifest_path), manifest)
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
