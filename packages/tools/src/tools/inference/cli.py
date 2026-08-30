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
    load_golden_classifier_scenes,
    load_golden_scenes,
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
    help="Train, eval, compare, sweep, export, accept, finalize, and fetch inference artifacts.",
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
        loss=loss,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        amp=True if amp else None,
        patience=patience,
        eval_interval=eval_interval,
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
    """Run the frozen-artifact acceptance gate.

    Without ``--scenes-dir`` the quality check has nothing to score, so the gate
    reports the manifest and contract results and then fails.
    """
    manifest = load_manifest(manifest_path)
    expected_in = (1, 4, height, width)
    try:
        if kind is InferenceKind.SEGMENTOR:
            report: AcceptanceReport | ClassifierAcceptanceReport = accept_artifact(
                artifact,
                manifest,
                scenes=(
                    load_golden_scenes(scenes_dir, scenes_split, scenes_limit) if scenes_dir else []
                ),
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
                scenes=(
                    load_golden_classifier_scenes(scenes_dir, scenes_split, scenes_limit)
                    if scenes_dir
                    else []
                ),
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


@app.command("finalize")
def finalize_command(
    run: Annotated[str, typer.Option(help="Training run directory.")],
    int8: Annotated[
        bool, typer.Option("--int8/--no-int8", help="Also export and accept INT8.")
    ] = True,
    calib_samples: Annotated[int, typer.Option(help="INT8 calibration samples.")] = 32,
    scenes_limit: Annotated[
        int, typer.Option("--scenes-limit", help="Maximum golden scenes; zero takes all.")
    ] = 0,
    min_iou: Annotated[float, typer.Option(help="Minimum mean mask IoU.")] = 0.5,
    min_accuracy: Annotated[float, typer.Option(help="Minimum classifier accuracy.")] = 0.9,
    max_latency_ms: Annotated[
        float, typer.Option(help="Maximum per-frame inference latency.")
    ] = 500.0,
    promote_path: Annotated[
        str | None, typer.Option("--promote", help="Copy the preferred accepted artifact.")
    ] = None,
) -> None:
    """Score test, export FP32 and INT8, and run the golden-scene gate."""
    from tools.inference.finalize import finalize

    try:
        report = finalize(
            run,
            int8=int8,
            calib_samples=calib_samples,
            scenes_limit=scenes_limit,
            min_iou=min_iou,
            min_accuracy=min_accuracy,
            max_latency_ms=max_latency_ms,
            promote_path=promote_path,
        )
    except (FileNotFoundError, ImportError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"fp32 {report.fp32_detail}")
    if report.int8_onnx:
        typer.echo(f"int8 {report.int8_detail}")
    if report.promoted:
        typer.echo(report.promoted)
    if not report.fp32_accepted and not report.int8_accepted:
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


@app.command("rank")
def rank_command(
    run_dir: Annotated[str, typer.Option(help="Parent directory of runs.")] = "artifacts/runs",
    metric: Annotated[str, typer.Option(help="Val metric to rank.")] = "mean_iou",
) -> None:
    """Print run summaries sorted by a val metric, then by FLOPs."""
    from tools.inference.runs import discover_runs, format_rank, rank_runs

    typer.echo(format_rank(rank_runs(discover_runs(run_dir), metric)), nl=False)


@app.command("pareto")
def pareto_command(
    run_dir: Annotated[str, typer.Option(help="Run catalog directory.")] = "artifacts/runs",
    metric: Annotated[str, typer.Option(help="Metric to maximize.")] = "mean_iou",
    cost: Annotated[str, typer.Option(help="Cost axis: n_params or flops.")] = "n_params",
    kind: Annotated[str, typer.Option(help="Restrict to one artifact kind.")] = "",
    split: Annotated[str, typer.Option(help="Score split: val or test.")] = "val",
    from_jsonl: Annotated[
        list[str] | None,
        typer.Option("--from-jsonl", help="Keep only run_ids recorded in this sweep JSONL."),
    ] = None,
    by_arch: Annotated[
        bool,
        typer.Option("--by-arch", help="Average extra seeds of the same architecture."),
    ] = False,
    baseline: Annotated[
        float | None,
        typer.Option(help="If set, print the knee and its neighbours, not the full frontier."),
    ] = None,
    spread: Annotated[
        float,
        typer.Option(help="Allowed drop below --baseline when picking the knee."),
    ] = 0.0,
    auto_spread: Annotated[
        bool,
        typer.Option(
            "--auto-spread",
            help="Set spread to the seed range of the first-pass knee architecture.",
        ),
    ] = False,
    neighbors: Annotated[
        int,
        typer.Option(help="Frontier neighbours on each side of the knee. Used with --baseline."),
    ] = 1,
    write_space: Annotated[
        list[str] | None,
        typer.Option(
            "--write-space",
            help="Replace the arch placeholder in this space TOML.",
        ),
    ] = None,
) -> None:
    """Print the runs on the size against quality frontier, cheapest first.

    Every point is read from one split so the comparison is like for like.
    ``--from-jsonl`` keeps one sweep; pass it more than once to join sweeps.
    ``--by-arch`` averages seeds. ``--baseline`` selects the cheapest holding
    point and its neighbours. ``--auto-spread`` uses the seed range at the knee.
    """
    from tools.inference.pareto import (
        format_pareto,
        frontier_points,
        knee,
        knee_neighbors,
        mean_by_arch,
        orient_score,
        pareto_front,
        score_spread,
        substitute_arch_placeholder,
    )
    from tools.inference.runs import discover_runs
    from tools.inference.sweep import completed_run_ids

    spaces = write_space or []
    if spaces and baseline is None:
        typer.echo("--write-space requires --baseline", err=True)
        raise typer.Exit(code=1)
    try:
        jsonl_paths = from_jsonl or []
        run_ids: frozenset[str] | None = None
        if jsonl_paths:
            collected: set[str] = set()
            for path in jsonl_paths:
                collected.update(completed_run_ids(path))
            run_ids = frozenset(collected)
        points = frontier_points(
            discover_runs(run_dir),
            metric,
            cost_key=cost,
            kind=kind,
            split=split,
            run_ids=run_ids,
        )
        raw_points = points
        if by_arch:
            points = mean_by_arch(points)
        front = pareto_front(points)
        if baseline is not None:
            oriented = orient_score(metric, baseline)
            used_spread = spread
            if auto_spread:
                first = knee(front, oriented, spread=spread)
                used_spread = max(spread, score_spread(raw_points, first.arch))
            selected = knee(front, oriented, spread=used_spread)
            front = knee_neighbors(front, selected, beside=neighbors)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(format_pareto(front, metric, cost), nl=False)
    if spaces:
        arches = tuple(point.arch for point in front)
        for dest in spaces:
            space = Path(dest)
            try:
                updated = substitute_arch_placeholder(space.read_text(encoding="utf-8"), arches)
            except (OSError, ValueError) as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(code=1) from exc
            space.write_text(updated, encoding="utf-8")
            typer.echo(str(space))


@app.command("sweep")
def sweep_command(
    space: Annotated[str, typer.Option(help="Sweep space TOML.")],
    out: Annotated[str | None, typer.Option(help="JSONL output path.")] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Append to the JSONL and skip trials it already records."),
    ] = False,
    data_dir: Annotated[
        str | None, typer.Option(help="Override the pack directory for every trial.")
    ] = None,
    run_dir: Annotated[
        str | None, typer.Option(help="Override the run parent directory for every trial.")
    ] = None,
) -> None:
    """Train a cartesian space, score val, and write sweep.jsonl."""
    from tools.inference.sweep import sweep

    try:
        path = sweep(space, out=out, resume=resume, data_dir=data_dir, run_dir=run_dir)
    except (ValueError, FileExistsError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(path)


@app.command("arches")
def arches_command() -> None:
    """Print registered kind and architecture name pairs."""
    from tools.inference.arch.registry import known

    for kind, name in sorted(known()):
        typer.echo(f"{kind}\t{name}")


def main(argv: list[str] | None = None) -> int:
    """Invoke the inference CLI and return its process exit code."""
    try:
        app(args=argv, prog_name="python -m tools.inference")
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return 0
