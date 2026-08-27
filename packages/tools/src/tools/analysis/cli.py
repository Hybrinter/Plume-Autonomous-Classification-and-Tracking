"""Command-line entry point: capture a scenario or suite and emit its report bundle.

Usage::

    python -m tools.analysis run <suite|scenario> [--out DIR]
    python -m tools.analysis list

``run`` drives the named suite (``full`` / ``builtin`` / ``files`` / a named grouping) or a single
scenario through the recorder and writes a per-run bundle (data/, figures/, summary.md + .html,
manifest.json) plus a suite index under ``--out`` (default ``artifacts/analysis/<name>``). ``list``
prints the available suites and scenarios. Everything is deterministic, so re-running reproduces an
identical bundle.

Contains:
  - app: package-owned Typer application.
  - main: invoke the application for console and ``python -m`` entry points.

Satisfies: REQ-OBS-SIL-001.
"""

from __future__ import annotations

from pathlib import Path

# third-party
import typer

# internal
from tools.analysis.characterize import run_suite, suite_names, suite_specs
from tools.analysis.datapoints import GROUPS, REGISTRY, accumulable_names
from tools.analysis.report import write_suite_report
from tools.analysis.runner import scenario_names

app = typer.Typer(
    help="Capture deterministic SIL runs and emit static telemetry report bundles.",
    no_args_is_help=True,
)


@app.command("run")
def run_command(
    name: str,
    out: str | None = typer.Option(
        default=None,
        help="Output directory (default: artifacts/analysis/<name>).",
    ),
) -> None:
    """Capture a suite or scenario and write its report bundle."""
    specs = suite_specs(name)
    runs = run_suite(name)
    out_dir = Path(out) if out is not None else Path("artifacts") / "analysis" / name
    report = write_suite_report(runs, name, out_dir)
    total_figures = sum(run.n_figures for run in report.runs)
    typer.echo(f"captured {len(runs)} scenario(s) for '{name}':")
    for spec, run_report in zip(specs, report.runs, strict=True):
        typer.echo(f"  - {spec.name:24s} {run_report.n_figures} figures -> {run_report.out_dir}")
    typer.echo(
        f"datapoints: {len(REGISTRY)} registry signals + {len(accumulable_names())} cumulative "
        f"across {len(GROUPS)} groups; {total_figures} figures total"
    )
    typer.echo(f"bundle written to {report.out_dir}")


@app.command("list")
def list_command() -> None:
    """List available suites and scenarios."""
    typer.echo("suites:")
    for name in suite_names():
        typer.echo(f"  {name}")
    typer.echo("scenarios:")
    for name in scenario_names():
        typer.echo(f"  {name}")


def main(argv: list[str] | None = None) -> int:
    """Invoke the analysis CLI and return its process exit code."""
    try:
        app(args=argv, prog_name="python -m tools.analysis")
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return 0
