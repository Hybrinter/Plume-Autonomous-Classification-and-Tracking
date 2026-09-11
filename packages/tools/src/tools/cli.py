"""Root command-line interface for PACT engineering tools.

Contains:
  - app: root Typer application with explicitly registered tools packages.
  - main: console-script entry point.
"""

from __future__ import annotations

# third-party
import typer

# internal
from tools.analysis.cli import app as analysis_app
from tools.inference.cli import app as inference_app

app = typer.Typer(
    help="PACT engineering utilities outside the flight image.",
    no_args_is_help=True,
)
app.add_typer(inference_app, name="inference")
app.add_typer(analysis_app, name="analysis")


def main(argv: list[str] | None = None) -> int:
    """Invoke the root tools CLI and return its process exit code."""
    try:
        app(args=argv, prog_name="pact-tools")
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return 0
