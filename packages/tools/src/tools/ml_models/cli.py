"""Command line for model dataset builds.

Contains:
  - app: Typer application mounted by the root tools CLI.
  - main: ``python -m tools.ml_models`` entry point.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from tools.ml_models.dataset.build import build_flight, build_synthetic, build_zenodo
from tools.ml_models.dataset.raw import BinSpec
from tools.ml_models.dataset.spec import BuildSpec, load_build_spec


class SourceName(StrEnum):
    """Raw sources the build command accepts."""

    FLIGHT = "flight"
    SYNTHETIC = "synthetic"
    ZENODO = "zenodo"


app = typer.Typer(
    help="Build finished model datasets.",
    no_args_is_help=True,
)
dataset_app = typer.Typer(
    help="Build a finished dataset from a raw tile source.",
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
        typer.Option(help="Zenodo GSD bin to emit (repeatable). Default: all bins."),
    ] = None,
    spec: Annotated[Path | None, typer.Option(help="Optional BuildSpec TOML.")] = None,
    n: Annotated[int, typer.Option(help="Synthetic tile count.")] = 12,
    seed: Annotated[int, typer.Option(help="Synthetic image seed.")] = 0,
) -> None:
    """Build a finished dataset from flight tiles, Zenodo archives, or the synthetic source."""
    try:
        resolved = load_build_spec(spec) if spec is not None else BuildSpec()
        if source is SourceName.FLIGHT:
            if source_dir is None:
                raise typer.BadParameter("--source flight requires --source-dir")
            build_flight(source_dir, out, resolved)
            return
        if source is SourceName.ZENODO:
            if images_tar is None or labels_tar is None or weights_path is None:
                raise typer.BadParameter(
                    "--source zenodo requires --images-tar, --labels-tar, and --weights-path"
                )
            build_zenodo(images_tar, labels_tar, weights_path, out, resolved, _select_bins(bin_id))
            return
        build_synthetic(out, resolved, n=n, seed=seed)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


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
