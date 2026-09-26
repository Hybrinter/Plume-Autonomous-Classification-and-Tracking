#!/usr/bin/env python3
"""Write a 76 px prism proxy pack from local Zenodo archives.

The script does not download archives. A missing weight table stops the run
before either archive is opened.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.ml_models.data.prism import write_prism_pack
from tools.ml_models.data.split import SplitRecipe

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_WEIGHTS = _REPO / "data" / "manifests" / "ap3200t_s2_weights.toml"


def _parser() -> argparse.ArgumentParser:
    """Return the command-line parser.

    Returns:
        argparse.ArgumentParser: Images, labels, weights, destination, and seed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Local image tar archive")
    parser.add_argument("--labels", type=Path, required=True, help="Local label tar archive")
    parser.add_argument(
        "--weights",
        type=Path,
        default=_DEFAULT_WEIGHTS,
        help="AP-3200T prism weight table",
    )
    parser.add_argument("--dest", type=Path, required=True, help="Output pack directory")
    parser.add_argument("--seed", type=int, default=0, help="Location-group split seed")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write one prism proxy pack.

    Args:
        argv: Arguments excluding the program name. ``None`` reads the process
            arguments.

    Returns:
        int: ``0`` after the pack is written.

    Raises:
        FileNotFoundError: If the weight table is missing. The archives are
            not opened in that case.
        ValueError: If the pack has no polygon tile or too few locations.
    """
    args = _parser().parse_args(argv)
    weights = Path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    write_prism_pack(
        args.images,
        args.labels,
        weights,
        args.dest,
        recipe=SplitRecipe(seed=int(args.seed)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
