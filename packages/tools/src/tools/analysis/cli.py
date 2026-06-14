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
  - main: parse argv and dispatch the ``run`` / ``list`` commands.

Satisfies: REQ-OBS-SIL-001.
"""

from __future__ import annotations

# stdlib
import argparse
from pathlib import Path

# internal
from tools.analysis.characterize import run_suite, suite_names, suite_specs
from tools.analysis.datapoints import GROUPS, REGISTRY, accumulable_names
from tools.analysis.report import write_suite_report
from tools.analysis.runner import scenario_names


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``run`` and ``list`` commands."""
    parser = argparse.ArgumentParser(
        prog="tools.analysis",
        description="Capture the deterministic SIL and emit a static telemetry report bundle.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="capture a suite/scenario and write a bundle")
    run_parser.add_argument(
        "name", help="suite (full/builtin/files/smoke/faults/commands/resources) or a scenario name"
    )
    run_parser.add_argument(
        "--out", default=None, help="output directory (default artifacts/analysis/<name>)"
    )
    subparsers.add_parser("list", help="list the available suites and scenarios")
    return parser


def _run(name: str, out: str | None) -> int:
    """Capture the named suite/scenario and write its bundle; return a process exit code."""
    specs = suite_specs(name)
    runs = run_suite(name)
    out_dir = Path(out) if out is not None else Path("artifacts") / "analysis" / name
    report = write_suite_report(runs, name, out_dir)
    total_figures = sum(run.n_figures for run in report.runs)
    print(f"captured {len(runs)} scenario(s) for '{name}':")
    for spec, run_report in zip(specs, report.runs, strict=True):
        print(f"  - {spec.name:24s} {run_report.n_figures} figures -> {run_report.out_dir}")
    print(
        f"datapoints: {len(REGISTRY)} registry signals + {len(accumulable_names())} cumulative "
        f"across {len(GROUPS)} groups; {total_figures} figures total"
    )
    print(f"bundle written to {report.out_dir}")
    return 0


def _list() -> int:
    """Print the available suites and scenarios; return a process exit code."""
    print("suites:")
    for name in suite_names():
        print(f"  {name}")
    print("scenarios:")
    for name in scenario_names():
        print(f"  {name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse argv and dispatch the requested command.

    Args:
        argv: optional argument vector (defaults to sys.argv[1:]).

    Returns:
        A process exit code (0 on success).
    """
    args = _build_parser().parse_args(argv)
    if args.command == "list":
        return _list()
    return _run(str(args.name), args.out if args.out is None else str(args.out))
