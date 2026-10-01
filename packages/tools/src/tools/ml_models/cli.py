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

from tools.ml_models.dataset.build import build_flight, build_synthetic
from tools.ml_models.dataset.spec import BuildSpec, load_build_spec


class SourceName(StrEnum):
    """Raw sources the build command accepts."""

    FLIGHT = "flight"
    SYNTHETIC = "synthetic"


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
    spec: Annotated[Path | None, typer.Option(help="Optional BuildSpec TOML.")] = None,
    n: Annotated[int, typer.Option(help="Synthetic tile count.")] = 12,
    seed: Annotated[int, typer.Option(help="Synthetic image seed.")] = 0,
) -> None:
    """Build a finished dataset from flight tiles or the synthetic source."""
    try:
        resolved = load_build_spec(spec) if spec is not None else BuildSpec()
        if source is SourceName.FLIGHT:
            if source_dir is None:
                raise typer.BadParameter("--source flight requires --source-dir")
            build_flight(source_dir, out, resolved)
            return
        build_synthetic(out, resolved, n=n, seed=seed)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


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
