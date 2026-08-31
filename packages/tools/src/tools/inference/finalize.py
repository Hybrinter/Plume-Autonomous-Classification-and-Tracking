"""Score test, export FP32 and INT8, and run the acceptance gate on a trained run.

Training and the sweep score the validation split. The numbers that admit an
artifact into ``data/models/`` are the test-split metrics, the exported ONNX
graphs, and the golden-scene gate. This module runs those three steps from the
run directory so a finalist does not depend on a hand-assembled command sequence.

Contains:
  - FinalizeReport: paths, gate outcomes, and the test-split eval path.
  - finalize: evaluate test, export, accept, and write ``finalize.json``.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tools.inference.accept import (
    AcceptanceReport,
    ClassifierAcceptanceReport,
    accept_kind,
    load_manifest,
)
from tools.inference.eval import evaluate
from tools.inference.export import ExportConfig, export, int8_artifact_path, promote
from tools.inference.train import load_train_config


@dataclass(frozen=True, slots=True)
class FinalizeReport:
    """Outcome of scoring, exporting, and gating one trained run.

    Attributes:
        run_dir: Training run directory.
        eval_path: Path of the written ``eval.json`` (test split).
        fp32_onnx: Exported FP32 artifact.
        fp32_accepted: Whether the FP32 gate passed.
        fp32_detail: FP32 gate detail string.
        int8_onnx: Exported INT8 artifact, empty when INT8 was not requested.
        int8_accepted: Whether the INT8 gate passed. False when INT8 was skipped.
        int8_detail: INT8 gate detail, or a skip reason.
        promoted: Destination of a promoted INT8 (preferred) or FP32 artifact.
    """

    run_dir: str
    eval_path: str
    fp32_onnx: str
    fp32_accepted: bool
    fp32_detail: str
    int8_onnx: str
    int8_accepted: bool
    int8_detail: str
    promoted: str


def _pack_dir(run_dir: Path, data_dir: str) -> Path:
    """Return the processed pack this run trained on."""
    if data_dir:
        candidate = Path(data_dir)
        if (candidate / "splits.json").is_file():
            return candidate
    synthetic = run_dir / "synthetic_pack"
    if (synthetic / "splits.json").is_file():
        return synthetic
    raise FileNotFoundError(f"no processed pack with splits.json for run {run_dir}")


GateOutcome = AcceptanceReport | ClassifierAcceptanceReport


def _accept(
    kind: str,
    artifact: Path,
    pack_dir: Path,
    height: int,
    width: int,
    min_iou: float,
    min_accuracy: float,
    max_latency_ms: float,
    scenes_limit: int,
) -> GateOutcome:
    """Run the matching golden-scene gate and return its report."""
    manifest = load_manifest(str(artifact.with_suffix(".json")))
    return accept_kind(
        kind,
        str(artifact),
        manifest,
        scenes_dir=str(pack_dir),
        scenes_split="test",
        scenes_limit=scenes_limit,
        expected_input=(1, 4, height, width),
        height=height,
        width=width,
        min_iou=min_iou,
        min_accuracy=min_accuracy,
        max_latency_ms=max_latency_ms,
    )


def finalize(
    run_dir: str | Path,
    *,
    int8: bool = True,
    calib_samples: int = 32,
    scenes_limit: int = 0,
    min_iou: float = 0.5,
    min_accuracy: float = 0.9,
    max_latency_ms: float = 500.0,
    promote_path: str | None = None,
) -> FinalizeReport:
    """Score the test split, export, accept, and write ``finalize.json``.

    Args:
        run_dir: Training run directory with ``config.toml`` and
            ``checkpoints/best.pt``.
        int8: Also export and accept a sibling INT8 artifact.
        calib_samples: INT8 calibration sample count from the train split.
        scenes_limit: Maximum golden scenes; ``0`` takes the whole test split.
        min_iou: Segmentor IoU floor.
        min_accuracy: Classifier accuracy floor.
        max_latency_ms: Worst-case per-scene latency budget.
        promote_path: When set, copy the preferred accepted artifact there.
            INT8 is preferred when it passed; otherwise FP32.

    Returns:
        FinalizeReport: Paths and gate outcomes. Also written as
        ``finalize.json`` in the run directory.

    Raises:
        FileNotFoundError: If the run, checkpoint, or pack is missing.
        ImportError: If the live ONNX gate cannot import onnxruntime.
    """
    root = Path(run_dir)
    cfg = load_train_config(str(root / "config.toml"))
    summary_path = root / "summary.json"
    summary: dict[str, object] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pack_dir = _pack_dir(root, cfg.data_dir)
    eval_path = evaluate(root, split="test")
    height = int(cfg.input_height_px)
    width = int(cfg.input_width_px)
    export_dir = root / "export"
    fp32_onnx = export_dir / f"{cfg.kind}.onnx"
    export(
        ExportConfig(
            kind=cfg.kind,
            checkpoint_path=str(root / "checkpoints" / "best.pt"),
            output_path=str(fp32_onnx),
            in_channels=int(cfg.in_channels),
            input_height_px=height,
            input_width_px=width,
            dataset_hash=str(summary.get("dataset_hash", "")),
            model_repo_sha=str(summary.get("model_repo_sha", "unknown")),
            int8=int8,
            calib_dir=str(pack_dir),
            calib_samples=calib_samples,
        )
    )
    fp32_report = _accept(
        cfg.kind,
        fp32_onnx,
        pack_dir,
        height,
        width,
        min_iou,
        min_accuracy,
        max_latency_ms,
        scenes_limit,
    )
    int8_onnx = ""
    int8_ok = False
    int8_detail = "skipped"
    int8_report: GateOutcome | None = None
    if int8:
        int8_path = int8_artifact_path(fp32_onnx)
        int8_onnx = str(int8_path)
        int8_report = _accept(
            cfg.kind,
            int8_path,
            pack_dir,
            height,
            width,
            min_iou,
            min_accuracy,
            max_latency_ms,
            scenes_limit,
        )
        int8_ok = int8_report.accepted
        int8_detail = int8_report.detail
    promoted = ""
    if promote_path is not None:
        chosen = int8_report if int8_ok and int8_report is not None else fp32_report
        if chosen.accepted:
            promoted = str(
                promote(str(Path(int8_onnx) if int8_ok else fp32_onnx), promote_path, chosen)
            )
    report = FinalizeReport(
        run_dir=str(root),
        eval_path=str(eval_path),
        fp32_onnx=str(fp32_onnx),
        fp32_accepted=fp32_report.accepted,
        fp32_detail=fp32_report.detail,
        int8_onnx=int8_onnx,
        int8_accepted=int8_ok,
        int8_detail=int8_detail,
        promoted=promoted,
    )
    (root / "finalize.json").write_text(
        json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8"
    )
    return report
