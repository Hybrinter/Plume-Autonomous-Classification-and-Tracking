"""Typer CLI for ml_models train, eval, export, accept, and pair.

Contains:
  - app: package-owned Typer application.
  - main: invoke the application and return a process exit code.
"""

from __future__ import annotations

from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from flight.libs.config import FaultConfig

from tools.ml_models.data.canvas import CanvasConfig
from tools.ml_models.train.config import (
    apply_train_mapping,
    load_train_config,
    overlay_train_config,
)


class ModelKind(StrEnum):
    """Supported artifact kinds."""

    CLASSIFIER = "classifier"
    SEGMENTOR = "segmentor"


app = typer.Typer(
    help="Train, eval, export, accept, and pair ml_models artifacts.",
    no_args_is_help=True,
)

_FAULT = FaultConfig()


@app.command("train")
def train_command(
    kind: Annotated[ModelKind | None, typer.Option(help="Artifact kind.")] = None,
    arch: Annotated[str | None, typer.Option(help="Architecture name.")] = None,
    config: Annotated[str | None, typer.Option(help="Optional TOML overlay.")] = None,
    data_dir: Annotated[str | None, typer.Option(help="Training data directory.")] = None,
    out: Annotated[str | None, typer.Option(help="Optional extra last.pt copy.")] = None,
    run_dir: Annotated[str | None, typer.Option(help="Parent directory for runs.")] = None,
    run_id: Annotated[str | None, typer.Option(help="Run directory name.")] = None,
    epochs: Annotated[int | None, typer.Option(help="Training epochs.")] = None,
    batch_size: Annotated[int | None, typer.Option(help="Training batch size.")] = None,
    height: Annotated[int | None, typer.Option(help="Input height in pixels.")] = None,
    width: Annotated[int | None, typer.Option(help="Input width in pixels.")] = None,
    seed: Annotated[int | None, typer.Option(help="Random seed.")] = None,
    in_channels: Annotated[int | None, typer.Option(help="Input band count.")] = None,
    learning_rate: Annotated[float | None, typer.Option(help="Optimizer learning rate.")] = None,
    momentum: Annotated[float | None, typer.Option(help="SGD momentum.")] = None,
    weight_decay: Annotated[float | None, typer.Option(help="Weight decay.")] = None,
    synthetic_samples: Annotated[int | None, typer.Option(help="Synthetic pack size.")] = None,
    bit_depth: Annotated[int | None, typer.Option(help="DN bit depth.")] = None,
    val_metric: Annotated[str | None, typer.Option(help="Best-checkpoint metric.")] = None,
    device: Annotated[str | None, typer.Option(help="Torch device string.")] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace an existing run directory.")
    ] = False,
    optimizer: Annotated[str | None, typer.Option(help="sgd or adamw.")] = None,
    scheduler: Annotated[str | None, typer.Option(help="none or cosine.")] = None,
    shuffle: Annotated[bool, typer.Option("--shuffle", help="Shuffle the train loader.")] = False,
    pos_weight: Annotated[float | None, typer.Option(help="Positive-class BCE weight.")] = None,
    augment: Annotated[
        bool, typer.Option("--augment", help="Flip and rotate the train split.")
    ] = False,
    loss: Annotated[
        str | None,
        typer.Option(help="bce, dice, bce_dice, focal, or focal_dice."),
    ] = None,
    focal_gamma: Annotated[float | None, typer.Option(help="Focal focusing exponent.")] = None,
    focal_alpha: Annotated[float | None, typer.Option(help="Focal positive-class weight.")] = None,
    amp: Annotated[bool, typer.Option("--amp", help="CUDA mixed-precision training.")] = False,
    patience: Annotated[
        int | None, typer.Option(help="Early-stop after this many unimproved scored epochs.")
    ] = None,
    eval_interval: Annotated[
        int | None, typer.Option(help="Epochs between scoring passes.")
    ] = None,
    max_steps: Annotated[int | None, typer.Option(help="Cap on optimizer steps.")] = None,
    canvas: Annotated[
        bool,
        typer.Option("--canvas", help="Train on CanvasConfig flight-frame defaults."),
    ] = False,
) -> None:
    """Train a classifier or segmentor and print the run directory."""
    from tools.ml_models.train.loop import train

    cfg = overlay_train_config(
        load_train_config(config),
        kind=kind.value if kind is not None else None,
        arch=arch,
        data_dir=data_dir,
        checkpoint_path=out,
        epochs=epochs,
        batch_size=batch_size,
        input_height_px=height,
        input_width_px=width,
        seed=seed,
        run_dir=run_dir,
        run_id=run_id,
        in_channels=in_channels,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        synthetic_samples=synthetic_samples,
        bit_depth=bit_depth,
        val_metric=val_metric,
        device=device,
        overwrite=True if overwrite else None,
        optimizer=optimizer,
        scheduler=scheduler,
        shuffle=True if shuffle else None,
        pos_weight=pos_weight,
        augment=True if augment else None,
        loss=loss,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        amp=True if amp else None,
        patience=patience,
        eval_interval=eval_interval,
        max_steps=max_steps,
    )
    if canvas:
        cfg = apply_train_mapping(cfg, {"canvas": asdict(CanvasConfig())})
    try:
        path = train(cfg)
    except (ValueError, FileExistsError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(path)


@app.command("eval")
def eval_command(
    run: Annotated[str, typer.Option(help="Run directory.")],
    split: Annotated[str, typer.Option(help="Split to score.")] = "val",
    checkpoint: Annotated[str | None, typer.Option(help="Checkpoint path.")] = None,
) -> None:
    """Score a checkpoint. The default split is val."""
    from tools.inference.eval import evaluate

    try:
        path = evaluate(run, checkpoint=checkpoint, split=split)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(path)


@app.command("export")
def export_command(
    run: Annotated[str, typer.Option(help="Training run directory.")],
    out: Annotated[str | None, typer.Option(help="ONNX output path.")] = None,
    checkpoint: Annotated[str | None, typer.Option(help="Checkpoint path.")] = None,
    height: Annotated[int | None, typer.Option(help="Fallback input height.")] = None,
    width: Annotated[int | None, typer.Option(help="Fallback input width.")] = None,
    int8: Annotated[
        bool, typer.Option("--int8", help="Also write a sibling INT8 QDQ ONNX file.")
    ] = False,
    fp16: Annotated[
        bool, typer.Option("--fp16", help="Also write a sibling FP16 ONNX file.")
    ] = False,
    override_spatial: Annotated[
        bool,
        typer.Option(
            "--override-spatial",
            help="Use --height and --width even when the checkpoint recorded a size.",
        ),
    ] = False,
    calib_dir: Annotated[
        str, typer.Option("--calib-dir", help="Processed pack for INT8 calibration.")
    ] = "",
    calib_samples: Annotated[
        int, typer.Option("--calib-samples", help="INT8 calibration sample count.")
    ] = 4,
) -> None:
    """Export ONNX logits and a sidecar. A flight-promotable run traces 1544 by 2064."""
    import json

    from tools.ml_models.export.onnx import (
        ExportConfig,
        export,
        fp16_artifact_path,
        int8_artifact_path,
    )
    from tools.ml_models.train.config import load_train_config

    root = Path(run)
    try:
        cfg = load_train_config(str(root / "config.toml"))
    except OSError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    ckpt = Path(checkpoint) if checkpoint is not None else root / "checkpoints" / "best.pt"
    dest = Path(out) if out is not None else root / "export" / f"{cfg.kind}.onnx"
    summary_path = root / "summary.json"
    dataset_hash = "synthetic"
    repo_sha = "unknown"
    if summary_path.is_file():
        raw = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            dataset_value = raw.get("dataset_hash", dataset_hash)
            repo_value = raw.get("model_repo_sha", repo_sha)
            if isinstance(dataset_value, str) and dataset_value:
                dataset_hash = dataset_value
            if isinstance(repo_value, str) and repo_value:
                repo_sha = repo_value
    export_height = int(cfg.input_height_px) if height is None else height
    export_width = int(cfg.input_width_px) if width is None else width
    export_cfg = ExportConfig(
        kind=cfg.kind,
        checkpoint_path=str(ckpt),
        output_path=str(dest),
        in_channels=int(cfg.in_channels),
        input_height_px=export_height,
        input_width_px=export_width,
        dataset_hash=dataset_hash,
        model_repo_sha=repo_sha,
        int8=int8,
        fp16=fp16,
        calib_dir=calib_dir,
        calib_samples=calib_samples,
        override_spatial=override_spatial,
    )
    try:
        onnx_path, manifest_path, _manifest = export(export_cfg)
    except (FileNotFoundError, ImportError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(onnx_path)
    typer.echo(manifest_path)
    if export_cfg.int8:
        int8_path = int8_artifact_path(onnx_path)
        typer.echo(int8_path)
        typer.echo(int8_path.with_suffix(".json"))
    if export_cfg.fp16:
        fp16_path = fp16_artifact_path(onnx_path)
        typer.echo(fp16_path)
        typer.echo(fp16_path.with_suffix(".json"))


@app.command("accept")
def accept_command(
    kind: Annotated[ModelKind, typer.Option(help="Artifact kind.")],
    artifact: Annotated[str, typer.Option(help="ONNX artifact path.")],
    manifest_path: Annotated[str, typer.Option("--manifest", help="Manifest JSON path.")],
    flight: Annotated[
        bool,
        typer.Option("--flight", help="Expect InferenceConfig input and output shapes."),
    ] = False,
    promote_path: Annotated[
        str | None, typer.Option("--promote", help="Destination path after a pass.")
    ] = None,
    min_iou: Annotated[float, typer.Option(help="Minimum mean mask IoU.")] = 0.5,
    min_accuracy: Annotated[float, typer.Option(help="Minimum classifier accuracy.")] = 0.9,
    max_latency_ms: Annotated[
        float, typer.Option(help="Maximum per-frame inference latency.")
    ] = _FAULT.inference_timeout_ms,
    scenes_dir: Annotated[
        str, typer.Option("--scenes-dir", help="Processed pack supplying golden scenes.")
    ] = "",
    scenes_split: Annotated[
        str, typer.Option("--scenes-split", help="Split read for golden scenes.")
    ] = "test",
    scenes_limit: Annotated[
        int, typer.Option("--scenes-limit", help="Maximum golden scenes; zero takes all.")
    ] = 0,
) -> None:
    """Run hash, I/O contract, and golden-scene checks."""
    from tools.ml_models.export.accept import accept_kind, load_manifest
    from tools.ml_models.export.onnx import promote

    manifest = load_manifest(manifest_path)
    expected_input: tuple[int | None, ...]
    if flight:
        expected_input = (1, 1)
        height = 1
        width = 1
    else:
        shape = manifest.input_shape
        if len(shape) < 4 or shape[2] is None or shape[3] is None:
            typer.echo(f"manifest input shape is not concrete NCHW: {shape}", err=True)
            raise typer.Exit(code=1)
        height = int(shape[2])
        width = int(shape[3])
        expected_input = shape
    try:
        report = accept_kind(
            kind.value,
            artifact,
            manifest,
            scenes_dir=scenes_dir,
            scenes_split=scenes_split,
            scenes_limit=scenes_limit,
            expected_input=expected_input,
            height=height,
            width=width,
            min_iou=min_iou,
            min_accuracy=min_accuracy,
            max_latency_ms=max_latency_ms,
            flight=flight,
        )
    except (ImportError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(report.detail)
    if report.accepted and promote_path is not None:
        typer.echo(promote(artifact, promote_path, report))
    if not report.accepted:
        raise typer.Exit(code=1)


@app.command("pair")
def pair_command(
    classifier: Annotated[str, typer.Option(help="Accepted classifier sidecar.")],
    segmentor: Annotated[str, typer.Option(help="Accepted segmentor sidecar.")],
    out: Annotated[str, typer.Option(help="Pair JSON destination.")],
) -> None:
    """Write the deploy pair blob. Raise when the flight shape does not match."""
    from tools.ml_models.export.pair import write_pair_manifest

    try:
        write_pair_manifest(classifier, segmentor, out)
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(out)


def main(argv: list[str] | None = None) -> int:
    """Invoke the ml_models CLI and return its process exit code."""
    try:
        app(args=argv, prog_name="python -m tools.ml_models")
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return 0
