"""Run the Orin Nano Super full-frame inference study.

    uv run python -m analysis.studies.orin_nano_full_frame_inference all
    uv run python -m analysis.studies.orin_nano_full_frame_inference bench

Contains:
  - main: argparse dispatch to write_all / run_ort_bench.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """Dispatch report generation or an optional ORT bench.

    Args:
        argv: Argument list without the program name. Default is sys.argv[1:].

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Orin Nano Super full-frame inference design study (not flight software)."
    )
    parser.add_argument(
        "command",
        choices=("all", "bench"),
        help="all: analytic figures and markdown. bench: optional ORT CSV.",
    )
    args = parser.parse_args(argv)
    if args.command == "all":
        from analysis.studies.orin_nano_full_frame_inference.report import write_all

        results, readme, figures = write_all()
        print(results)
        print(readme)
        for path in figures:
            print(path)
    else:
        from analysis.studies.orin_nano_full_frame_inference.latency import run_ort_bench

        written = run_ort_bench()
        if written is None:
            print("ORT bench skipped (onnxruntime or artifacts missing)")
            return 0
        print(written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
