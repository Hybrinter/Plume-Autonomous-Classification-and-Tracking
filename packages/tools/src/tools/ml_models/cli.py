"""Command line for model dataset, training, export, and analysis workflows.

Contains:
  - app: Typer application mounted by the root tools CLI.
  - main: ``python -m tools.ml_models`` entry point.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from tools.ml_models.dataset.build import build_flight, build_zenodo
from tools.ml_models.dataset.raw import BinSpec
from tools.ml_models.dataset.spec import BuildSpec, load_build_spec


class SourceName(StrEnum):
    """Raw sources the build command accepts."""

    FLIGHT = "flight"
    ZENODO = "zenodo"


app = typer.Typer(
    help="Build finished model datasets.",
    no_args_is_help=True,
)
dataset_app = typer.Typer(
    help="Build or analyze a finished dataset.",
    no_args_is_help=True,
)
app.add_typer(dataset_app, name="dataset")


@dataset_app.command("build")
def build_command(
    source: Annotated[SourceName, typer.Option(..., help="Raw source kind.")],
    out: Annotated[Path, typer.Option(..., help="Finished dataset directory.")],
    source_dir: Annotated[
        Path | None,
        typer.Option(help="Flight tile directory. Required for --source flight."),
    ] = None,
    images_tar: Annotated[
        Path | None,
        typer.Option(help="Zenodo image archive. Required for --source zenodo."),
    ] = None,
    labels_tar: Annotated[
        Path | None,
        typer.Option(help="Zenodo annotation archive. Required for --source zenodo."),
    ] = None,
    weights_path: Annotated[
        Path | None,
        typer.Option(help="Prism weight table TOML. Required for --source zenodo."),
    ] = None,
    bin_id: Annotated[
        list[str] | None,
        typer.Option(
            help="Zenodo GSD bin name (repeatable): native10 or gsd15-gsd35. Default: all bins."
        ),
    ] = None,
    spec: Annotated[Path | None, typer.Option(help="Optional BuildSpec TOML.")] = None,
) -> None:
    """Build a finished dataset from flight tiles or Zenodo archives."""
    try:
        resolved = load_build_spec(spec) if spec is not None else BuildSpec()
        if source is SourceName.FLIGHT:
            if source_dir is None:
                raise typer.BadParameter("--source flight requires --source-dir")
            build_flight(source_dir, out, resolved)
            return
        if images_tar is None or labels_tar is None or weights_path is None:
            raise typer.BadParameter(
                "--source zenodo requires --images-tar, --labels-tar, and --weights-path"
            )
        build_zenodo(images_tar, labels_tar, weights_path, out, resolved, _select_bins(bin_id))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@dataset_app.command("analyze")
def dataset_analyze_command(
    dataset: Annotated[Path, typer.Option(..., help="Finished dataset directory.")],
    out: Annotated[Path, typer.Option(..., help="Analysis output directory.")],
) -> None:
    """Measure one finished dataset into an evidence bundle.

    The numerical measurement and publication API is implemented, but the
    CLI surface is unavailable until dataset rendering is implemented.
    """
    raise typer.BadParameter(
        "dataset analysis CLI is unavailable until dataset rendering is implemented"
    )


@app.command("train")
def train_command(
    config: Annotated[
        Path | None,
        typer.Option(help="TrainConfig TOML. Defaults apply when omitted."),
    ] = None,
    kind: Annotated[
        str | None,
        typer.Option(help="Model task: classifier or segmentor."),
    ] = None,
    arch: Annotated[
        str | None,
        typer.Option(help="Architecture grammar name. Empty selects the kind default."),
    ] = None,
    dataset: Annotated[
        list[str] | None,
        typer.Option(help="Finished dataset directory. Exactly one is accepted."),
    ] = None,
    run_dir: Annotated[
        str | None,
        typer.Option(help="Run root directory."),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option(help="Run directory name under --run-dir."),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option(help="Torch device. Default: cuda when available else cpu."),
    ] = None,
    epochs: Annotated[int | None, typer.Option(help="Epoch count.")] = None,
    batch_size: Annotated[int | None, typer.Option(help="Rows per batch.")] = None,
    max_steps: Annotated[
        int | None,
        typer.Option(help="Global optimizer-step cap."),
    ] = None,
) -> None:
    """Train a GSD-conditioned model on one finished dataset."""
    from flight.libs.types import Err

    from tools.ml_models.train.config import apply_train_mapping, load_train_config
    from tools.ml_models.train.loop import train

    if dataset is not None and len(dataset) > 1:
        raise typer.BadParameter("train takes exactly one --dataset")
    overlay: dict[str, object] = {}
    for key, value in (
        ("kind", kind),
        ("arch", arch),
        ("run_dir", run_dir),
        ("run_id", run_id),
        ("device", device),
        ("epochs", epochs),
        ("batch_size", batch_size),
        ("max_steps", max_steps),
    ):
        if value is not None:
            overlay[key] = value
    if dataset is not None:
        overlay["dataset"] = dataset[0]
    try:
        cfg = apply_train_mapping(
            load_train_config(str(config) if config is not None else None), overlay
        )
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    result = train(cfg)
    if isinstance(result, Err):
        raise typer.BadParameter(result.error)
    typer.echo(str(result.value))


@app.command("export")
def export_command(
    checkpoint: Annotated[
        Path,
        typer.Option(..., help="Trained conditioned checkpoint (.pt)."),
    ],
    out: Annotated[
        Path,
        typer.Option(..., help="Destination ONNX artifact path."),
    ],
    allow_partial_gsd: Annotated[
        bool,
        typer.Option(help="Record a GSD coverage gap instead of rejecting it."),
    ] = False,
    dynamic_spatial: Annotated[
        bool,
        typer.Option(help="Allow varying image sizes for research exports."),
    ] = False,
) -> None:
    """Export a conditioned checkpoint as a validated two-input ONNX artifact."""
    from flight.libs.types import Err

    from tools.ml_models.export.export import ExportConfig, export

    result = export(
        ExportConfig(
            checkpoint_path=str(checkpoint),
            output_path=str(out),
            allow_partial_gsd=allow_partial_gsd,
            dynamic_spatial=dynamic_spatial,
        )
    )
    if isinstance(result, Err):
        raise typer.BadParameter(result.error)
    typer.echo(str(result.value))


@app.command("accept")
def accept_command(
    artifact: Annotated[Path, typer.Option(..., help="ONNX artifact to gate.")],
    manifest: Annotated[
        Path,
        typer.Option(..., help="Model sidecar JSON for the artifact."),
    ],
    dataset: Annotated[
        list[str],
        typer.Option(help="Finished dataset directory. Exactly one is accepted."),
    ],
    min_iou: Annotated[
        float, typer.Option(help="Minimum per-source mean IoU for segmentors.")
    ] = 0.5,
    min_accuracy: Annotated[
        float, typer.Option(help="Minimum per-source accuracy for classifiers.")
    ] = 0.9,
    max_latency_ms: Annotated[
        float,
        typer.Option(help="Worst allowed batch-one CPU latency in milliseconds."),
    ] = 20.0,
) -> None:
    """Gate an artifact on the finished test split and write an acceptance report."""
    import json

    from flight.libs.types import Err

    from tools.ml_models.export.accept import accept_artifact
    from tools.ml_models.export.manifest import acceptance_path, load_manifest

    if len(dataset) != 1:
        raise typer.BadParameter("accept takes exactly one --dataset")
    report_path = acceptance_path(artifact)
    if report_path.exists():
        raise typer.BadParameter(f"refusing to overwrite {report_path}")
    try:
        model_manifest = load_manifest(manifest)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    result = accept_artifact(
        artifact,
        model_manifest,
        dataset[0],
        min_iou=min_iou,
        min_accuracy=min_accuracy,
        max_latency_ms=max_latency_ms,
    )
    if isinstance(result, Err):
        raise typer.BadParameter(result.error)
    report = result.value
    evaluation = report["evaluation"]
    assert isinstance(evaluation, dict)
    payload = {
        "sha256": model_manifest.sha256,
        "dataset": str(dataset[0]),
        "dataset_hash": str(evaluation["dataset_hash"]),
        "min_iou": min_iou,
        "min_accuracy": min_accuracy,
        **report,
    }
    try:
        with report_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2) + "\n")
    except OSError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(str(report_path))
    if not report["accepted"]:
        raise typer.Exit(code=1)


@app.command("pair")
def pair_command(
    classifier_sidecar: Annotated[Path, typer.Option(..., help="Classifier model sidecar JSON.")],
    segmentor_sidecar: Annotated[Path, typer.Option(..., help="Segmentor model sidecar JSON.")],
    out: Annotated[Path, typer.Option(..., help="Destination pair manifest JSON.")],
    allow_partial_gsd: Annotated[
        bool,
        typer.Option(help="Record a GSD coverage gap instead of rejecting it."),
    ] = False,
) -> None:
    """Write a combined manifest for an accepted classifier/segmentor pair."""
    from flight.libs.types import Err

    from tools.ml_models.export.pair import write_pair_manifest

    result = write_pair_manifest(
        classifier_sidecar, segmentor_sidecar, out, allow_partial_gsd=allow_partial_gsd
    )
    if isinstance(result, Err):
        raise typer.BadParameter(result.error)
    typer.echo(str(out))


class Precision(StrEnum):
    """Artifact precision conversions the convert command accepts."""

    FP16 = "fp16"
    INT8 = "int8"


@app.command("convert")
def convert_command(
    precision: Annotated[
        Precision,
        typer.Option(..., help="Target precision: fp16 graph weights or int8 QDQ."),
    ],
    source: Annotated[
        Path,
        typer.Option(..., help="Source FP32 ONNX artifact with a valid sidecar."),
    ],
    out: Annotated[Path, typer.Option(..., help="New destination artifact path.")],
    dataset: Annotated[
        list[str] | None,
        typer.Option(help="Finished dataset directory for INT8 calibration. At most one."),
    ] = None,
    calib_samples: Annotated[int, typer.Option(help="Maximum INT8 calibration batches.")] = 32,
) -> None:
    """Convert a validated FP32 artifact to FP16 or INT8 with a new sidecar."""
    from flight.libs.types import Err

    from tools.ml_models.export.precision import convert_fp16, quantize_int8

    if dataset is not None and len(dataset) > 1:
        raise typer.BadParameter("convert takes at most one --dataset")
    if precision == Precision.INT8:
        if dataset is None or len(dataset) != 1:
            raise typer.BadParameter("int8 conversion requires exactly one --dataset")
        result = quantize_int8(source, out, dataset=dataset[0], calib_samples=calib_samples)
    else:
        result = convert_fp16(source, out)
    if isinstance(result, Err):
        raise typer.BadParameter(result.error)
    typer.echo(str(result.value))


@app.command("analyze")
def analyze_command(
    run: Annotated[Path, typer.Option(..., help="Training run directory to analyze.")],
    out: Annotated[Path, typer.Option(..., help="Analysis output directory.")],
    checkpoint: Annotated[str, typer.Option(help="Checkpoint selector. Default: best.")] = "best",
    final_test: Annotated[bool, typer.Option(help="Include the final-test evaluation.")] = False,
) -> None:
    """Analyze one training run. Unavailable until model evidence lands."""
    from flight.libs.types import Err

    from tools.ml_models.analysis.config import ModelAnalysisConfig
    from tools.ml_models.analysis.model import analyze_model

    result = analyze_model(
        ModelAnalysisConfig(
            run=str(run), out=str(out), checkpoint=checkpoint, final_test=final_test
        )
    )
    if isinstance(result, Err):
        raise typer.BadParameter(result.error)
    typer.echo(str(result.value))


@app.command("render")
def render_command(
    evidence: Annotated[Path, typer.Option(..., help="Frozen evidence directory.")],
    out: Annotated[Path, typer.Option(..., help="Destination figure directory.")],
) -> None:
    """Render figures from frozen evidence. Unavailable until plotting lands."""
    from flight.libs.types import Err

    from tools.ml_models.analysis.config import PlotConfig
    from tools.ml_models.analysis.plots.common import render_analysis

    result = render_analysis(evidence, PlotConfig(), out)
    if isinstance(result, Err):
        raise typer.BadParameter(result.error)
    typer.echo(str(result.value))


def _select_bins(bin_ids: list[str] | None) -> tuple[BinSpec, ...] | None:
    """Resolve repeatable ``--bin-id`` names against ``DEFAULT_BINS``.

    Args:
        bin_ids: Names passed on the command line. None selects every bin.

    Returns:
        tuple[BinSpec, ...] | None: The named bins in ``DEFAULT_BINS`` order,
        or None for the full default table.

    Raises:
        typer.BadParameter: If a name is unknown or repeated.
    """
    if bin_ids is None:
        return None
    from tools.ml_models.dataset.sources.zenodo.bins import DEFAULT_BINS

    if len(set(bin_ids)) != len(bin_ids):
        raise typer.BadParameter(f"duplicate --bin-id values {sorted(bin_ids)}")
    by_name = {item.bin_id: item for item in DEFAULT_BINS}
    unknown = sorted(set(bin_ids) - set(by_name))
    if unknown:
        raise typer.BadParameter(f"unknown --bin-id {unknown}; known bins {sorted(by_name)}")
    selected = set(bin_ids)
    return tuple(item for item in DEFAULT_BINS if item.bin_id in selected)


def main(argv: list[str] | None = None) -> int:
    """Invoke the model CLI and return its process exit code.

    Args:
        argv: Argument vector without the program name. None reads ``sys.argv``.

    Returns:
        int: Process exit code.
    """
    try:
        app(args=argv, prog_name="tools.ml_models")
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 1
    return 0
