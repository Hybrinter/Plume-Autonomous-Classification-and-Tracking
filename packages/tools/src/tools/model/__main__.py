"""CLI for tools.model: ``python -m tools.model <train|export|accept>``.

Train and export are scaffolds in this layer and exit with status 2.
Accept is a library API in this layer; the CLI reports that and exits 2
until the accept subcommand is wired.

Contains:
  - main: parse subcommands and dispatch.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """Parse argv and dispatch a tools.model subcommand.

    Args:
        argv: Argument list without the program name. None reads sys.argv[1:].

    Returns:
        int: Process exit code. 0 is unused in this scaffold; 2 means the
        subcommand is not implemented in this layer.
    """
    parser = argparse.ArgumentParser(prog="python -m tools.model")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="train classifier or segmentor")
    sub.add_parser("export", help="export frozen ONNX artifacts")
    sub.add_parser("accept", help="run the artifact acceptance gate")
    args = parser.parse_args(argv)
    print(
        f"tools.model {args.command} is not implemented as a CLI in this layer; "
        "use the Python API (tools.model.accept) for acceptance.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
