"""Command line for the VIS/SWIR fusion trade.

Contains:
  - main: dispatch the ``test``, ``late``, ``thresholds``, and ``seed-plot`` commands.

``test`` and ``late`` run inference and need a GPU-sized machine; do not run
them while a training job is running. ``thresholds`` reads the capture rows of
a finished model analysis. ``seed-plot`` reads only JSON written by ``test``
and ``late``: each file holds a ``test`` object of metric name to value.

Satisfies: REQ-AIML-COMP-001.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from flight.libs.types import Err

from tools.ml_models.analysis.config import ModelAnalysisConfig, load_model_analysis_config
from tools.ml_models.fusion.late import DEFAULT_VIS_WEIGHTS
from tools.ml_models.fusion.plots import write_seed_plot, write_threshold_plot
from tools.ml_models.fusion.study import late_fusion, load_run, score_split
from tools.ml_models.fusion.thresholds import (
    choose_thresholds,
    read_capture_rows,
    score_thresholds,
)


def _parser() -> argparse.ArgumentParser:
    """Return the argument parser with one sub-parser per command."""
    parser = argparse.ArgumentParser(prog="python -m tools.ml_models.fusion")
    sub = parser.add_subparsers(dest="command", required=True)

    test = sub.add_parser("test", help="score one run's selected checkpoint on test")
    test.add_argument("--config", type=Path, required=True, help="model analysis TOML")
    test.add_argument("--run", type=Path, required=True, help="run directory")
    test.add_argument("--dataset", type=Path, help="evaluation dataset; default training set")
    test.add_argument("--out", type=Path, required=True, help="output JSON")

    late = sub.add_parser("late", help="late fusion: VIS weight on val, then test")
    late.add_argument("--config", type=Path, required=True, help="model analysis TOML")
    late.add_argument("--vis-run", type=Path, required=True, help="VIS-only run")
    late.add_argument("--swir-run", type=Path, required=True, help="SWIR-only run")
    late.add_argument("--reference-run", type=Path, required=True, help="any run of the same task")
    late.add_argument("--dataset", type=Path, required=True, help="paired evaluation dataset")
    late.add_argument("--metric", required=True, help="validation metric to maximize")
    late.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=list(DEFAULT_VIS_WEIGHTS),
        help="VIS weights to search (default 0 to 1 in 0.1 steps)",
    )
    late.add_argument("--out", type=Path, required=True, help="output JSON")

    thr = sub.add_parser("thresholds", help="val-chosen thresholds scored on test")
    thr.add_argument("--analysis", type=Path, required=True, help="model analysis directory")
    thr.add_argument("--out", type=Path, required=True, help="output JSON")
    thr.add_argument("--plot", type=Path, help="optional output PNG")
    thr.add_argument("--title", default="Decision threshold", help="plot title prefix")

    seeds = sub.add_parser("seed-plot", help="per-seed test scores for each option")
    seeds.add_argument(
        "--entry",
        action="append",
        required=True,
        help="LABEL=a.json,b.json (repeat once per option, in display order)",
    )
    seeds.add_argument("--metric", default="average_precision", help="test metric name")
    seeds.add_argument("--label", default="Test average precision", help="y-axis label")
    seeds.add_argument("--title", default="Score per training seed", help="figure title")
    seeds.add_argument("--out", type=Path, required=True, help="output PNG")
    return parser


def _write_json(path: Path, payload: object) -> None:
    """Write indented JSON, creating the parent directory.

    Args:
        path: Output file.
        payload: JSON-serializable value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def _config(path: Path) -> ModelAnalysisConfig:
    """Load the analysis settings or exit with the error.

    Args:
        path: Model analysis TOML.

    Returns:
        ModelAnalysisConfig: Parsed settings.

    Raises:
        SystemExit: If the file is not a valid configuration.
    """
    loaded = load_model_analysis_config(path)
    if isinstance(loaded, Err):
        raise SystemExit(loaded.error)
    return loaded.value


def _run_test(args: argparse.Namespace) -> int:
    """Score one run on test and write ``{"run", "dataset", "test"}``."""
    base = _config(args.config)
    inputs = load_run(base, args.run, args.dataset)
    if isinstance(inputs, Err):
        raise SystemExit(inputs.error)
    scored = score_split(inputs.value.model, inputs.value, base, "test")
    if isinstance(scored, Err):
        raise SystemExit(scored.error)
    dataset = None if args.dataset is None else str(args.dataset)
    _write_json(args.out, {"run": str(args.run), "dataset": dataset, "test": scored.value})
    print(f"{args.run.name}: {json.dumps({k: round(v, 4) for k, v in scored.value.items()})}")
    return 0


def _run_late(args: argparse.Namespace) -> int:
    """Run the late-fusion search and write the chosen weight and test metrics."""
    base = _config(args.config)
    vis = load_run(base, args.vis_run, None)
    if isinstance(vis, Err):
        raise SystemExit(vis.error)
    swir = load_run(base, args.swir_run, None)
    if isinstance(swir, Err):
        raise SystemExit(swir.error)
    paired = load_run(base, args.reference_run, args.dataset)
    if isinstance(paired, Err):
        raise SystemExit(paired.error)
    result = late_fusion(
        vis.value, swir.value, paired.value, base, args.metric, tuple(args.weights)
    )
    if isinstance(result, Err):
        raise SystemExit(result.error)
    out = result.value
    _write_json(
        args.out,
        {
            "vis_run": str(args.vis_run),
            "swir_run": str(args.swir_run),
            "dataset": str(args.dataset),
            "metric": args.metric,
            "vis_weight": out.vis_weight,
            "val_by_weight": {f"{w:.2f}": v for w, v in out.val_by_weight.items()},
            "test": out.test,
        },
    )
    print(f"{args.vis_run.name} + {args.swir_run.name}: w={out.vis_weight:.1f}")
    return 0


def _run_thresholds(args: argparse.Namespace) -> int:
    """Choose thresholds on validation capture rows and score them on test."""
    capture = args.analysis / "capture"
    choice = choose_thresholds(read_capture_rows(capture / "val" / "rows.jsonl"))
    rows = score_thresholds(read_capture_rows(capture / "test" / "rows.jsonl"), choice)
    _write_json(
        args.out,
        {
            "analysis": str(args.analysis),
            "global_logit": choice.global_logit,
            "per_bin_logit": choice.per_bin_logit,
            "rows": [asdict(row) for row in rows],
        },
    )
    if args.plot is not None:
        write_threshold_plot(rows, args.title, args.plot)
    for row in rows:
        print(
            f"{row.bin_id:6} n={row.n:5d} fixed {row.f1_fixed:.3f} "
            f"global {row.f1_global:.3f} per-bin {row.f1_per_bin:.3f}"
        )
    return 0


def _run_seed_plot(args: argparse.Namespace) -> int:
    """Read per-seed test JSON files and write the seed plot."""
    scores: dict[str, list[float]] = {}
    for entry in args.entry:
        label, sep, files = entry.partition("=")
        if not sep or not files:
            raise SystemExit(f"--entry must be LABEL=a.json,b.json, got {entry!r}")
        values: list[float] = []
        for name in files.split(","):
            payload = json.loads(Path(name).read_text(encoding="utf-8"))
            if args.metric not in payload.get("test", {}):
                raise SystemExit(f"{name} has no test {args.metric}")
            values.append(float(payload["test"][args.metric]))
        scores[label] = values
    write_seed_plot(scores, args.label, args.title, args.out)
    for label, values in scores.items():
        print(f"{label}: {', '.join(f'{v:.4f}' for v in values)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run one fusion-trade command.

    Args:
        argv: Arguments excluding the program name. ``None`` reads the process
            arguments.

    Returns:
        int: Zero on success.

    Raises:
        SystemExit: On a bad argument, a load error, or an evaluation error.
    """
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    match args.command:
        case "test":
            return _run_test(args)
        case "late":
            return _run_late(args)
        case "thresholds":
            return _run_thresholds(args)
        case _:
            return _run_seed_plot(args)
