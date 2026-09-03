"""Run the single-axis vs dual-axis gimbal study.

    uv run python -m analysis.studies.single_axis_vs_dual_axis_gimbal geometry
    uv run python -m analysis.studies.single_axis_vs_dual_axis_gimbal industry
    uv run python -m analysis.studies.single_axis_vs_dual_axis_gimbal all

Contains:
  - main: argparse dispatch to run_geometry / run_industry.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """Dispatch geometry, industry, or both.

    Args:
        argv: Argument list without the program name. Default is sys.argv[1:].

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Single-axis vs dual-axis gimbal design study (not flight software)."
    )
    parser.add_argument(
        "command",
        choices=("geometry", "industry", "all"),
        help="geometry: design-pass T(lat, R). industry: world stack inventory. all: both.",
    )
    args = parser.parse_args(argv)
    if args.command in {"geometry", "all"}:
        from analysis.studies.single_axis_vs_dual_axis_gimbal.geometry import run_geometry

        run_geometry()
    if args.command in {"industry", "all"}:
        from analysis.studies.single_axis_vs_dual_axis_gimbal.expected import run_industry

        run_industry()
    return 0


if __name__ == "__main__":
    sys.exit(main())
