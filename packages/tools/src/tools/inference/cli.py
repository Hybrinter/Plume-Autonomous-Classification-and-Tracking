"""Typer CLI for inference training, export, acceptance, fetch, eval, and compare.

Contains:
  - app: package-owned Typer application.
  - main: invoke the application for console and ``python -m`` entry points.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

# stdlib
from enum import StrEnum
from pathlib import Path
from typing import Annotated

# third-party
import typer

# internal
from tools.inference.accept import (
    AcceptanceReport,
    ClassifierAcceptanceReport,
    accept_artifact,
    accept_classifier_artifact,
    load_manifest,
    onnx_classifier_inference_fn,
    onnx_inference_fn,
)
from tools.inference.export import ExportConfig, export, int8_artifact_path, promote
from tools.inference.fetch import main as fetch_main
from tools.inference.train import load_train_config, overlay_train_config, train


class InferenceKind(StrEnum):
    """Supported inference artifact kinds."""

    CLASSIFIER = "classifier"
    SEGMENTOR = "segmentor"


app = typer.Typer(
    help="Train, export, accept, and fetch data for flight inference artifacts.",
    no_args_is_help=True,
)


@app.command("train")
def train_command(
    kind: Annotated[InferenceKind | None, typer.Option(help="Artifact kind.")] = None,
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
) -> None:
    """Train a classifier or segmentor and write a run directory."""
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
    )
    try:
        path = train(cfg)
    except (ValueError, FileExistsError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(path)


@app.command("export")
def export_command(
    kind: Annotated[InferenceKind, typer.Option(help="Artifact kind.")],
    checkpoint: Annotated[str, typer.Option(help="Checkpoint path.")],
    out: Annotated[str, typer.Option(help="ONNX output path.")],
    height: Annotated[int, typer.Option(help="Input height in pixels.")] = 256,
    width: Annotated[int, typer.Option(help="Input width in pixels.")] = 256,
    version: Annotated[str, typer.Option(help="Artifact version.")] = "v1",
    dataset_hash: Annotated[str, typer.Option(help="Training dataset digest.")] = "synthetic",
    repo_sha: Annotated[str, typer.Option(help="Source repository revision.")] = "unknown",
    int8: Annotated[
        bool, typer.Option("--int8", help="Also write a sibling INT8 QDQ ONNX file.")
    ] = False,
    calib_dir: Annotated[
        str, typer.Option("--calib-dir", help="Processed pack for INT8 calibration.")
    ] = "",
    calib_samples: Annotated[
        int, typer.Option("--calib-samples", help="INT8 calibration sample count.")
    ] = 4,
) -> None:
    """Export a frozen ONNX artifact and manifest."""
    config = ExportConfig(
        kind=kind.value,
        checkpoint_path=checkpoint,
        output_path=out,
        input_height_px=height,
        input_width_px=width,
        version=version,
        dataset_hash=dataset_hash,
        model_repo_sha=repo_sha,
        int8=int8,
        calib_dir=calib_dir,
        calib_samples=calib_samples,
    )
    try:
        onnx_path, manifest_path, _manifest = export(config)
    except ImportError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(onnx_path)
    typer.echo(manifest_path)
    if config.int8:
        int8_path = int8_artifact_path(onnx_path)
        typer.echo(int8_path)
        typer.echo(int8_path.with_suffix(".json"))


@app.command("accept")
def accept_command(
    kind: Annotated[InferenceKind, typer.Option(help="Artifact kind.")],
    artifact: Annotated[str, typer.Option(help="ONNX artifact path.")],
    manifest_path: Annotated[str, typer.Option("--manifest", help="Manifest JSON path.")],
    promote_path: Annotated[
        str | None, typer.Option("--promote", help="Destination path after a pass.")
    ] = None,
    min_iou: Annotated[float, typer.Option(help="Minimum mean mask IoU.")] = 0.5,
    min_accuracy: Annotated[float, typer.Option(help="Minimum classifier accuracy.")] = 0.9,
    max_latency_ms: Annotated[
        float, typer.Option(help="Maximum per-frame inference latency.")
    ] = 500.0,
    height: Annotated[int, typer.Option(help="Input height in pixels.")] = 256,
    width: Annotated[int, typer.Option(help="Input width in pixels.")] = 256,
) -> None:
    """Run the frozen-artifact acceptance gate."""
    manifest = load_manifest(manifest_path)
    expected_in = (1, 4, height, width)
    try:
        if kind is InferenceKind.SEGMENTOR:
            report: AcceptanceReport | ClassifierAcceptanceReport = accept_artifact(
                artifact,
                manifest,
                scenes=[],
                run_inference=onnx_inference_fn(artifact),
                expected_input=expected_in,
                expected_output=(1, 1, height, width),
                min_iou=min_iou,
                max_latency_ms=max_latency_ms,
            )
        else:
            report = accept_classifier_artifact(
                artifact,
                manifest,
                scenes=[],
                run_inference=onnx_classifier_inference_fn(artifact),
                expected_input=expected_in,
                expected_output=(1, 1),
                min_accuracy=min_accuracy,
                max_latency_ms=max_latency_ms,
            )
    except ImportError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(report.detail)
    if report.accepted and promote_path is not None:
        typer.echo(promote(artifact, promote_path, report))
    if not report.accepted:
        raise typer.Exit(code=1)


@app.command(
    "fetch",
    context_settings={"allow_extra_args": False, "ignore_unknown_options": False},
)
def fetch_command(
    manifest: Annotated[str | None, typer.Option(help="Checksum manifest TOML.")] = None,
    raw_dir: Annotated[str | None, typer.Option(help="Raw dataset directory.")] = None,
    processed_dir: Annotated[str | None, typer.Option(help="Processed dataset directory.")] = None,
    download: Annotated[bool, typer.Option(help="Fetch missing or mismatched files.")] = False,
    preprocess: Annotated[bool, typer.Option(help="Write four-band PACT tensors.")] = False,
    height: Annotated[int, typer.Option(help="Processed image height.")] = 256,
    width: Annotated[int, typer.Option(help="Processed image width.")] = 256,
    limit: Annotated[int, typer.Option(help="Maximum samples; zero means all.")] = 0,
    split_recipe: Annotated[str | None, typer.Option(help="Split recipe TOML.")] = None,
) -> None:
    """Inspect, download, or preprocess the smoke-plume dataset."""
    argv: list[str] = []
    if manifest is not None:
        argv.extend(("--manifest", manifest))
    if raw_dir is not None:
        argv.extend(("--raw-dir", raw_dir))
    if processed_dir is not None:
        argv.extend(("--processed-dir", processed_dir))
    if download:
        argv.append("--download")
    if preprocess:
        argv.append("--preprocess")
    argv.extend(("--height", str(height), "--width", str(width), "--limit", str(limit)))
    if split_recipe is not None:
        argv.extend(("--split-recipe", split_recipe))
    code = fetch_main(argv)
    if code != 0:
        raise typer.Exit(code=code)


@app.command("eval")
def eval_command(
    run: Annotated[str, typer.Option(help="Run directory.")],
    checkpoint: Annotated[str | None, typer.Option(help="Checkpoint path.")] = None,
    split: Annotated[str, typer.Option(help="Split to score.")] = "val",
) -> None:
    """Score a checkpoint on a held-out split."""
    from tools.inference.eval import evaluate

    path = evaluate(run, checkpoint=checkpoint, split=split)
    typer.echo(path)


@app.command("report")
def report_command(
    run: Annotated[str, typer.Option(help="Run directory.")],
) -> None:
    """Write figures and report.md into a run directory."""
    from tools.inference.report import write_report

    typer.echo(write_report(run))


@app.command("list")
def list_command(
    run_dir: Annotated[str, typer.Option(help="Parent directory of runs.")] = "artifacts/runs",
) -> None:
    """Print a table of local run directories."""
    from tools.inference.runs import discover_runs, format_list

    typer.echo(format_list(discover_runs(run_dir)), nl=False)


@app.command("compare")
def compare_command(
    run: Annotated[list[str], typer.Option(help="Run directory (repeatable).")],
) -> None:
    """Print a side-by-side table of run summaries."""
    from tools.inference.runs import format_compare

    typer.echo(format_compare(tuple(Path(item) for item in run)), nl=False)


def main(argv: list[str] | None = None) -> int:
    """Invoke the inference CLI and return its process exit code."""
    try:
        app(args=argv, prog_name="python -m tools.inference")
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return 0
