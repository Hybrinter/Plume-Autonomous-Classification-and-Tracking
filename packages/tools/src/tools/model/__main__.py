"""CLI for tools.model: ``python -m tools.model <train|export|accept>``.

Train writes a checkpoint. Export writes an ONNX graph and a JSON manifest.
Accept runs the intake gate. Promote copies a passed artifact into data/models/.

Contains:
  - main: parse subcommands and dispatch.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import argparse
import sys

from tools.model.accept import (
    AcceptanceReport,
    ClassifierAcceptanceReport,
    accept_artifact,
    accept_classifier_artifact,
    load_manifest,
    onnx_classifier_inference_fn,
    onnx_inference_fn,
)
from tools.model.export import ExportConfig, export, promote
from tools.model.train import load_train_config, overlay_train_config, train


def _add_train_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the train subcommand."""
    parser = sub.add_parser("train", help="train classifier or segmentor")
    parser.add_argument("--kind", choices=("classifier", "segmentor"), default=None)
    parser.add_argument("--config", default=None, help="optional TOML overlay")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default=None, help="checkpoint path")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)


def _add_export_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the export subcommand."""
    parser = sub.add_parser("export", help="export frozen ONNX artifacts")
    parser.add_argument("--kind", choices=("classifier", "segmentor"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, help="onnx output path")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--dataset-hash", default="synthetic")
    parser.add_argument("--repo-sha", default="unknown")


def _add_accept_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the accept subcommand."""
    parser = sub.add_parser("accept", help="run the artifact acceptance gate")
    parser.add_argument("--kind", choices=("classifier", "segmentor"), required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--promote", default=None, help="destination path after a pass")
    parser.add_argument("--min-iou", type=float, default=0.5)
    parser.add_argument("--min-accuracy", type=float, default=0.9)
    parser.add_argument("--max-latency-ms", type=float, default=500.0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)


def _train_from_args(args: argparse.Namespace) -> int:
    """Dispatch train from parsed CLI args."""
    cfg = overlay_train_config(
        load_train_config(args.config),
        kind=args.kind,
        data_dir=args.data_dir,
        checkpoint_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        input_height_px=args.height,
        input_width_px=args.width,
        seed=args.seed,
    )
    try:
        path = train(cfg)
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(path)
    return 0


def _export_from_args(args: argparse.Namespace) -> int:
    """Dispatch export from parsed CLI args."""
    config = ExportConfig(
        kind=args.kind,
        checkpoint_path=args.checkpoint,
        output_path=args.out,
        input_height_px=args.height,
        input_width_px=args.width,
        version=args.version,
        dataset_hash=args.dataset_hash,
        model_repo_sha=args.repo_sha,
    )
    try:
        onnx_path, manifest_path, _manifest = export(config)
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(onnx_path)
    print(manifest_path)
    return 0


def _accept_from_args(args: argparse.Namespace) -> int:
    """Dispatch accept from parsed CLI args. Quality uses an empty scene list."""
    manifest = load_manifest(args.manifest)
    expected_in = (1, 4, args.height, args.width)
    try:
        if args.kind == "segmentor":
            report: AcceptanceReport | ClassifierAcceptanceReport = accept_artifact(
                args.artifact,
                manifest,
                scenes=[],
                run_inference=onnx_inference_fn(args.artifact),
                expected_input=expected_in,
                expected_output=(1, 1, args.height, args.width),
                min_iou=args.min_iou,
                max_latency_ms=args.max_latency_ms,
            )
        else:
            report = accept_classifier_artifact(
                args.artifact,
                manifest,
                scenes=[],
                run_inference=onnx_classifier_inference_fn(args.artifact),
                expected_input=expected_in,
                expected_output=(1, 1),
                min_accuracy=args.min_accuracy,
                max_latency_ms=args.max_latency_ms,
            )
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(report.detail)
    if report.accepted and args.promote:
        dest = promote(args.artifact, args.promote, report)
        print(dest)
    return 0 if report.accepted else 1


def main(argv: list[str] | None = None) -> int:
    """Parse argv and dispatch a tools.model subcommand.

    Args:
        argv: Argument list without the program name. None reads sys.argv[1:].

    Returns:
        int: 0 on success, 1 on gate failure or missing extra, argparse
        SystemExit on bad argv.
    """
    parser = argparse.ArgumentParser(prog="python -m tools.model")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_train_parser(sub)
    _add_export_parser(sub)
    _add_accept_parser(sub)
    args = parser.parse_args(argv)
    if args.command == "train":
        return _train_from_args(args)
    if args.command == "export":
        return _export_from_args(args)
    if args.command == "accept":
        return _accept_from_args(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
