#!/usr/bin/env python3
"""Score full-frame scenes from a processed pack.

``--dry-run`` builds scenes with ``sample_view`` and scores fixed logits.
The default frame is 1544 by 2064. ``--frame-h`` and ``--frame-w`` override
that size. The written JSON reports hit rate by placement, the empty-frame
false-positive rate, and the chip-versus-frame logit margin. ``chip_iou`` may
be present and is not a pass or fail field.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.ml_models.analysis.full_frame import (
    build_eval_scenes,
    score_dry_run,
    summarize_frames,
)


def _parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(prog="full_frame_eval")
    parser.add_argument("--pack", type=Path, help="Processed pack with a test split.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/full_frame/summary.json"),
    )
    parser.add_argument("--frame-h", type=int, default=None, help="Frame height. Default 1544.")
    parser.add_argument("--frame-w", type=int, default=None, help="Frame width. Default 2064.")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build scenes and score fixed logits.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write the full-frame JSON summary.

    Args:
        argv: Arguments excluding the program name. ``None`` reads the process
            arguments.

    Returns:
        int: 0 after the summary is written. 2 when ``--pack`` is omitted or
        ``--dry-run`` is omitted.
    """
    args = _parser().parse_args(argv)
    if args.pack is None:
        print("Pass --pack with a processed pack.", file=sys.stderr)
        return 2
    if not args.dry_run:
        print("Pass --dry-run to score scenes without weights.", file=sys.stderr)
        return 2
    scenes = build_eval_scenes(
        args.pack,
        limit=args.limit,
        seed=args.seed,
        frame_h=args.frame_h,
        frame_w=args.frame_w,
    )
    summary = summarize_frames(tuple(score_dry_run(scene) for scene in scenes))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
