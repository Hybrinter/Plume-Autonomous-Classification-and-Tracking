#!/usr/bin/env python3
"""Train ground-sample cells with the ml_models loop and score the test split.

Each cell calls ``tools.ml_models.train.loop.train`` once and then
``evaluate(..., split="test")`` once. The sweep does not export. Classifier
cells use ``shufflenetv2_x0_5``. Segmentor cells use ``dilatenet``. The loss
is ``bce``. Images are rewritten with ``band_z`` before training.

With no ``--pack`` and no Zenodo cache, the process prints a message and
exits 2. It does not download a corpus.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np
from tools.ml_models.analysis.eval import evaluate
from tools.ml_models.data.bands import ZENODO_BAND_IDS, verify_band_order
from tools.ml_models.data.matrix import Cell, gsd_cells, native_cells
from tools.ml_models.data.norm import apply_band_z, fit_band_stats
from tools.ml_models.train.config import TrainConfig
from tools.ml_models.train.loop import train
from tools.ml_models.train.recipe import DatasetMeta, compute_dataset_hash, write_dataset_meta

TrainKind = Literal["classifier", "segmentor"]

_CACHE_CANDIDATES = (
    Path("data/zenodo/cache"),
    Path("artifacts/zenodo/cache"),
)


def train_kind(task: str) -> TrainKind:
    """Map a matrix task onto a train kind.

    Args:
        task: ``classify`` or ``segment``.

    Returns:
        TrainKind: ``classifier`` or ``segmentor``.

    Raises:
        ValueError: If ``task`` is unknown.
    """
    match task:
        case "classify":
            return "classifier"
        case "segment":
            return "segmentor"
        case _:
            raise ValueError(f"unknown cell task {task!r}")


def train_arch(kind: TrainKind) -> str:
    """Return the study architecture for one train kind.

    Args:
        kind: ``classifier`` or ``segmentor``.

    Returns:
        str: ``shufflenetv2_x0_5`` or ``dilatenet``.

    Raises:
        ValueError: If ``kind`` is unknown.
    """
    match kind:
        case "classifier":
            return "shufflenetv2_x0_5"
        case "segmentor":
            return "dilatenet"
        case _:
            raise ValueError(f"unknown train kind {kind!r}")


def find_cache(explicit: Path | None = None) -> Path | None:
    """Return a Zenodo cache directory, or None when it is absent.

    Args:
        explicit: Optional cache directory. It counts when ``meta.json`` and
            ``stacks.dat`` are present.

    Returns:
        Path | None: The cache directory, or None.
    """
    candidates = (explicit,) if explicit is not None else _CACHE_CANDIDATES
    for path in candidates:
        if path is None:
            continue
        if (path / "meta.json").is_file() and (path / "stacks.dat").is_file():
            return path
    return None


def prepare_band_z_pack(src: Path, dest: Path) -> Path:
    """Write a copy of ``src`` whose images are train-split ``band_z``.

    Args:
        src: Processed pack with ``images.npy``, ``masks.npy``, ``labels.npy``,
            and ``splits.json``.
        dest: Destination directory.

    Returns:
        Path: ``dest``.

    Raises:
        ValueError: If the train split is empty.
    """
    images = np.asarray(np.load(src / "images.npy"), dtype=np.float32)
    masks = np.asarray(np.load(src / "masks.npy"), dtype=np.float32)
    labels = np.asarray(np.load(src / "labels.npy"), dtype=np.float32)
    splits = json.loads((src / "splits.json").read_text(encoding="utf-8"))
    train_rows = [int(index) for index in splits["train"]]
    if not train_rows:
        raise ValueError("train split is empty")
    stats = fit_band_stats(images[train_rows])
    scaled = apply_band_z(images, stats)
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "images.npy", np.ascontiguousarray(scaled))
    np.save(dest / "masks.npy", np.ascontiguousarray(masks))
    np.save(dest / "labels.npy", np.ascontiguousarray(labels))
    (dest / "splits.json").write_text((src / "splits.json").read_text(), encoding="utf-8")
    (dest / "provenance.json").write_text(
        json.dumps({"norm": "band_z"}) + "\n",
        encoding="utf-8",
    )
    digest = compute_dataset_hash(dest)
    meta = DatasetMeta(
        dataset_hash=digest,
        source_doi="",
        n=int(scaled.shape[0]),
        height=int(scaled.shape[2]),
        width=int(scaled.shape[3]),
        in_channels=int(scaled.shape[1]),
    )
    write_dataset_meta(dest / "dataset.json", meta)
    return dest


def run_cell(
    cell: Cell,
    pack_dir: str | Path,
    out_dir: str | Path,
    *,
    epochs: int = 1,
    max_steps: int | None = None,
) -> Path:
    """Train one cell and score its test split.

    Args:
        cell: Matrix cell. ``classify`` maps to ``classifier`` and
            ``segment`` maps to ``segmentor``.
        pack_dir: Processed pack. A ``band_z`` copy is written under ``out_dir``.
        out_dir: Parent directory for the pack copy and the run.
        epochs: Training epochs.
        max_steps: Optimizer-step cap. ``None`` runs every step in ``epochs``.

    Returns:
        Path: Run directory from ``train``. ``eval.json`` is the test split.

    Raises:
        ValueError: If the cell task is unknown.
    """
    kind = train_kind(cell.task)
    arch = train_arch(kind)
    root = Path(out_dir)
    prepared = prepare_band_z_pack(
        Path(pack_dir),
        root / "band_z" / f"{cell.task}-{cell.subset}-{cell.side_px}",
    )
    image_shape = np.load(prepared / "images.npy", mmap_mode="r").shape
    cfg = TrainConfig(
        kind=kind,
        arch=arch,
        loss="bce",
        epochs=epochs,
        max_steps=max_steps,
        batch_size=1,
        data_dir=str(prepared),
        input_height_px=int(image_shape[2]),
        input_width_px=int(image_shape[3]),
        in_channels=int(image_shape[1]),
        run_dir=str(root / "runs"),
        run_id=f"{cell.task}-{cell.subset}-{cell.side_px}",
        device="cpu",
        overwrite=True,
    )
    run_dir = train(cfg)
    evaluate(run_dir, split="test")
    return run_dir


def _cells(axis: str, limit: int | None) -> tuple[Cell, ...]:
    """Return native or ground-sample cells, truncated by ``limit``."""
    order = verify_band_order(ZENODO_BAND_IDS)
    planned = gsd_cells(order) if axis == "gsd" else native_cells(order)
    if limit is None:
        return planned
    if limit < 1:
        raise ValueError(f"limit must be >= 1; got {limit}")
    return planned[:limit]


def _parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(prog="gsd_sweep")
    parser.add_argument("--pack", type=Path, help="Processed pack. Skips the Zenodo cache.")
    parser.add_argument("--out", type=Path, default=Path("artifacts/gsd"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of cells.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--axis", choices=("gsd", "native"), default="gsd")
    parser.add_argument("--cache", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Train the requested cells, or exit when the cache and pack are absent.

    Args:
        argv: Arguments excluding the program name. ``None`` reads the process
            arguments.

    Returns:
        int: 0 after the cells finish. 2 when no pack is given and the Zenodo
        cache is absent, or when a cache is present and no pack was passed.
    """
    args = _parser().parse_args(argv)
    if args.pack is None:
        cache = find_cache(args.cache)
        if cache is None:
            print(
                "Zenodo cache is absent. Pass --pack with a processed pack. "
                "This command does not download.",
                file=sys.stderr,
            )
            return 2
        print(
            f"Zenodo cache is at {cache}. Pass --pack with a processed pack. "
            "This command does not download.",
            file=sys.stderr,
        )
        return 2
    for cell in _cells(args.axis, args.limit):
        run_dir = run_cell(
            cell,
            args.pack,
            args.out,
            epochs=args.epochs,
            max_steps=args.max_steps,
        )
        print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
