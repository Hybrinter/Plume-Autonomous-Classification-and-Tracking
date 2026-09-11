"""Run the analysis-only tracking estimator selection checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analysis.studies.tracking_estimator.report import write_selection_checkpoint


def main(argv: list[str] | None = None) -> int:
    """Write the selection checkpoint markdown to a caller-selected destination."""
    parser = argparse.ArgumentParser(description="Run the elevation estimator comparison study.")
    parser.add_argument(
        "--out", type=Path, required=True, help="Selection-checkpoint markdown path."
    )
    args = parser.parse_args(argv)
    print(write_selection_checkpoint(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
