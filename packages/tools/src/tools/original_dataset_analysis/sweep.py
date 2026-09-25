"""Native-resolution training sweep.

Contains:
  - CellResult: one trained cell, its metrics, and its loss history.
  - run_native: train the requested native cells.
  - write_review: tables, metric charts, and loss curves.
  - train_native: command line for the native sweep.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from tools.original_dataset_analysis.bands import (
    BandOrder,
    BandSpec,
    BandSubset,
    coerce_descriptions,
    resolve_subset,
    verify_band_order,
)
from tools.original_dataset_analysis.cache import (
    SidePack,
    TileCache,
    build_cache,
    build_mask_cache,
    open_cache,
    open_side_pack,
    prepare_side,
)
from tools.original_dataset_analysis.dataset import StudyDataset
from tools.original_dataset_analysis.grid import rasterize_mask
from tools.original_dataset_analysis.index import TileIndex, TileRef, build_index
from tools.original_dataset_analysis.matrix import Cell, gsd_cells, native_cells
from tools.original_dataset_analysis.metrics import score_classifier, score_segmentor
from tools.original_dataset_analysis.models import DilateNet, ShuffleNetClassifier
from tools.original_dataset_analysis.normalize import BandStats, MomentAccumulator
from tools.original_dataset_analysis.plots import write_delta_bars, write_loss_curve, write_pr_curve
from tools.original_dataset_analysis.results import write_filled_tables
from tools.original_dataset_analysis.split import SplitRecipe, assign_location_splits
from tools.original_dataset_analysis.train import EpochLoss, TrainConfig, run_training

_BATCH = 64
_CLASSIFY_METRICS = ("precision", "recall", "f1", "pr_auc", "roc_auc")
_SEGMENT_FIGURES = ("mean_iou", "dice")


@dataclass(frozen=True, slots=True)
class CellResult:
    """Scores and history for one native cell.

    Attributes:
        task: ``classify`` or ``segment``.
        subset: Subset name.
        side_px: Output side.
        best_epoch: Zero-based epoch of the best validation loss.
        metrics: Test-split metrics for the selected weights.
        history: Per-epoch train, validation, and test loss.
    """

    task: str
    subset: str
    side_px: int
    best_epoch: int
    metrics: Mapping[str, float]
    history: tuple[EpochLoss, ...]
    pr_recall: tuple[float, ...] = ()
    pr_precision: tuple[float, ...] = ()

    @property
    def score(self) -> float:
        """Return PR-AUC or Dice."""
        key = "pr_auc" if self.task == "classify" else "dice"
        return float(self.metrics[key])


def _subset(order: BandOrder, name: str) -> BandSubset:
    """Resolve one matrix subset name."""
    if name == "loo_" or name.startswith("loo_"):
        dropped = name.removeprefix("loo_")
        return resolve_subset(order, BandSpec("loo", dropped_band=dropped))
    if name in {"rgb", "s2_12"}:
        return resolve_subset(order, BandSpec(name))
    raise ValueError(f"unknown subset {name!r}")


def _tiles(index: TileIndex, stems: Sequence[str]) -> tuple[TileRef, ...]:
    """Return tiles in archive order for the given stems."""
    wanted = set(stems)
    return tuple(tile for tile in index.tiles if tile.stem in wanted)


def _stats(
    tiles: Sequence[TileRef],
    cache: TileCache | None,
    subset: BandSubset,
    *,
    pack: SidePack | None = None,
    rows: np.ndarray | None = None,
) -> BandStats:
    """Fit train moments on the selected channels."""
    accumulator = MomentAccumulator(len(subset.indices))
    columns = list(subset.indices)
    if pack is not None and rows is not None:
        for row in rows:
            accumulator.update(np.asarray(pack.images[int(row)][columns], dtype=np.float32))
        return accumulator.finish()
    if cache is None:
        raise ValueError("stats need a cache or a pack")
    for tile in tiles:
        accumulator.update(cache.reader(tile)[columns])
    return accumulator.finish()


def _loader(
    tiles: Sequence[TileRef],
    cache: TileCache,
    subset: BandSubset,
    stats: BandStats,
    side_px: int,
    *,
    shuffle: bool,
    seed: int,
    device: torch.device,
    cached_masks: np.ndarray | None = None,
    mask_rows: Mapping[str, int] | None = None,
    pack: SidePack | None = None,
    rows: np.ndarray | None = None,
) -> DataLoader[tuple[torch.Tensor, ...]]:
    """Return a loader over one split."""
    if pack is not None and rows is not None:
        dataset = StudyDataset.from_pack(pack, rows, subset.indices, stats)
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=_BATCH,
            shuffle=shuffle,
            generator=generator if shuffle else None,
            num_workers=0,
            pin_memory=False,
        )
    dataset = StudyDataset(
        tiles,
        cache.reader,
        subset,
        stats,
        side_px,
        cached_masks=cached_masks,
        mask_rows=mask_rows,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=_BATCH,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def _metrics(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    device: torch.device,
    target: str,
) -> tuple[dict[str, float], tuple[float, ...], tuple[float, ...]]:
    """Score the test loader with the weights already loaded in ``model``."""
    model.eval()
    logit_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            image, label, mask, annotated = batch
            if target == "mask":
                keep = annotated.reshape(-1) > 0
                if int(keep.sum().item()) == 0:
                    continue
                image, label, mask = image[keep], label[keep], mask[keep]
            image = image.to(device)
            logits = model(image)
            if target == "label":
                logit_parts.append(logits.reshape(-1).detach().cpu())
                target_parts.append(label.reshape(-1).detach().cpu())
            else:
                logit_parts.append(logits.detach().cpu())
                target_parts.append(mask.detach().cpu())
    if not logit_parts:
        raise ValueError("test loader has no usable samples")
    logits = torch.cat(logit_parts, dim=0)
    truth = torch.cat(target_parts, dim=0)
    if target == "label":
        scored = score_classifier(logits, truth)
        raw = logits.detach().double().reshape(-1).cpu().numpy()
        truth_np = truth.detach().double().reshape(-1).cpu().numpy() >= 0.5
        from tools.original_dataset_analysis.metrics import _ranking_areas

        _pr, _roc, recall_curve, precision_curve = _ranking_areas(raw, truth_np.astype(float))
        del _pr, _roc
        return (
            {
                "precision": scored.precision,
                "recall": scored.recall,
                "f1": scored.f1,
                "pr_auc": scored.pr_auc,
                "roc_auc": scored.roc_auc,
            },
            tuple(float(value) for value in recall_curve),
            tuple(float(value) for value in precision_curve),
        )
    scored_mask = score_segmentor(logits, truth)
    return (
        {
            "mean_iou": scored_mask.mean_iou,
            "dice": scored_mask.dice,
            "accuracy": scored_mask.accuracy,
        },
        (),
        (),
    )


def _model(task: str, channels: int) -> tuple[nn.Module, str]:
    """Return the head and the training target for one cell."""
    if task == "classify":
        return ShuffleNetClassifier(channels), "label"
    if task == "segment":
        return DilateNet(channels), "mask"
    raise ValueError(f"unknown task {task!r}")


def run_native(
    index: TileIndex,
    cache: TileCache,
    order: BandOrder,
    cells: Sequence[Cell],
    config: TrainConfig,
    device: torch.device,
    *,
    cached_masks: np.ndarray | None = None,
    mask_rows: Mapping[str, int] | None = None,
    mask_by_side: Mapping[int, np.ndarray] | None = None,
    pack: SidePack | None = None,
    on_result: Callable[[tuple[CellResult, ...]], None] | None = None,
) -> tuple[CellResult, ...]:
    """Train each native cell and score its best validation weights.

    Args:
        index: Corpus index.
        cache: Native stacks.
        order: Verified band order.
        cells: Cells to train. Ground-sample sides other than 120 are refused.
        config: Shared loop settings.
        device: Torch device.
        cached_masks: Native mask planes, or ``None`` to rasterize per sample.
        mask_rows: Stem to row in ``cached_masks``.

    Returns:
        tuple[CellResult, ...]: One result per cell, in input order.

    Raises:
        ValueError: If a cell is not native resolution.
    """
    split = assign_location_splits(index, SplitRecipe(seed=config.seed))
    grouped = {name: _tiles(index, stems) for name, stems in split.stems.items()}
    stem_row = None if pack is None else {stem: row for row, stem in enumerate(pack.stems)}
    results: list[CellResult] = []
    for cell in cells:
        if cell.side_px not in {120, 80, 60, 40, 30}:
            raise ValueError(f"side must be a legal side; got {cell.side_px}")
        print(f"training {cell.task} {cell.subset} {cell.side_px}", flush=True)
        side_masks = cached_masks
        if mask_by_side is not None and cell.side_px in mask_by_side:
            side_masks = mask_by_side[cell.side_px]
        subset = _subset(order, cell.subset)
        train_rows = None
        val_rows = None
        test_rows = None
        if stem_row is not None:
            train_rows = np.asarray(
                [stem_row[tile.stem] for tile in grouped["train"]], dtype=np.int32
            )
            val_rows = np.asarray([stem_row[tile.stem] for tile in grouped["val"]], dtype=np.int32)
            test_rows = np.asarray(
                [stem_row[tile.stem] for tile in grouped["test"]], dtype=np.int32
            )
        stats = _stats(
            grouped["train"],
            None if pack is not None else cache,
            subset,
            pack=pack,
            rows=train_rows,
        )
        model, target = _model(cell.task, len(subset.indices))
        trained = run_training(
            model,
            _loader(
                grouped["train"],
                cache,
                subset,
                stats,
                cell.side_px,
                shuffle=True,
                seed=config.seed,
                device=device,
                cached_masks=side_masks,
                mask_rows=mask_rows,
                pack=pack,
                rows=train_rows,
            ),
            _loader(
                grouped["val"],
                cache,
                subset,
                stats,
                cell.side_px,
                shuffle=False,
                seed=config.seed,
                device=device,
                cached_masks=side_masks,
                mask_rows=mask_rows,
                pack=pack,
                rows=val_rows,
            ),
            config,
            target=target,
            device=device,
            test_loader=_loader(
                grouped["test"],
                cache,
                subset,
                stats,
                cell.side_px,
                shuffle=False,
                seed=config.seed,
                device=device,
                cached_masks=side_masks,
                mask_rows=mask_rows,
                pack=pack,
                rows=test_rows,
            ),
        )
        model.load_state_dict(trained.state_dict)
        metrics, pr_recall, pr_precision = _metrics(
            model,
            _loader(
                grouped["test"],
                cache,
                subset,
                stats,
                cell.side_px,
                shuffle=False,
                seed=config.seed,
                device=device,
                cached_masks=side_masks,
                mask_rows=mask_rows,
                pack=pack,
                rows=test_rows,
            ),
            device,
            target,
        )
        results.append(
            CellResult(
                task=cell.task,
                subset=cell.subset,
                side_px=cell.side_px,
                best_epoch=trained.best_epoch,
                metrics=metrics,
                history=trained.history,
                pr_recall=pr_recall,
                pr_precision=pr_precision,
            )
        )
        if on_result is not None:
            on_result(tuple(results))
    return tuple(results)


def write_mask_previews(cache: TileCache, tiles: Sequence[TileRef], path: Path, limit: int) -> None:
    """Write a few RGB tiles with the rasterized smoke mask overlaid.

    Args:
        cache: Native stacks.
        tiles: Corpus tiles. Only tiles with polygons are drawn.
        path: Directory for the PNG files.
        limit: Maximum number of overlays.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = [tile for tile in tiles if tile.polygons][:limit]
    path.mkdir(parents=True, exist_ok=True)
    for tile in chosen:
        stack = cache.reader(tile)
        if stack.shape[0] >= 4:
            view = stack[[1, 2, 3]]
        else:
            view = np.repeat(stack[:1], 3, axis=0)
        image = np.moveaxis(view, 0, -1)
        low = np.percentile(image, 2, axis=(0, 1), keepdims=True)
        high = np.percentile(image, 98, axis=(0, 1), keepdims=True)
        stretched = np.clip((image - low) / np.maximum(high - low, 1e-6), 0.0, 1.0)
        polygons = tile.polygons if tile.polygons is not None else ()
        mask = rasterize_mask(polygons, 120, rule="half")[0]
        figure, axis = plt.subplots()
        axis.imshow(stretched)
        axis.imshow(np.ma.masked_where(mask < 0.5, mask), cmap="autumn", alpha=0.45)
        axis.set_axis_off()
        safe = "".join("-" if char in '<>:"/\\|?*' else char for char in tile.stem)
        figure.savefig(path / f"{safe}.png")
        plt.close(figure)


def write_review(
    order: BandOrder,
    results: Sequence[CellResult],
    path: Path,
    *,
    prevalence: Mapping[str, float] | None = None,
) -> None:
    """Write the score table, one chart per metric, and one loss curve per cell.

    Args:
        order: Verified band order.
        results: Trained cells. Ground-sample scores stay blank.
        path: Review directory. ``RESULTS.md``, ``results.json``, and
            ``figures/`` are created.
    """
    path.mkdir(parents=True, exist_ok=True)
    scores = {(item.task, item.subset, item.side_px): item.score for item in results}
    write_filled_tables(order, scores, path / "RESULTS.md", prevalence=prevalence)
    payload = [
        {
            "task": item.task,
            "subset": item.subset,
            "side_px": item.side_px,
            "score": item.score,
            "best_epoch": item.best_epoch,
            "metrics": dict(item.metrics),
            "history": [
                {
                    "epoch": row.epoch,
                    "train_loss": row.train_loss,
                    "val_loss": row.val_loss,
                    "test_loss": row.test_loss,
                }
                for row in item.history
            ],
        }
        for item in results
    ]
    (path / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    figures = path / "figures"
    for task, metric_names in (("classify", _CLASSIFY_METRICS), ("segment", _SEGMENT_FIGURES)):
        rows = [item for item in results if item.task == task and item.side_px == 120]
        if not rows:
            continue
        baseline = next((item for item in rows if item.subset == "s2_12"), None)
        if baseline is None:
            continue
        for metric in metric_names:
            write_delta_bars(
                f"{task} {metric} minus s2_12",
                [item.subset for item in rows],
                [float(item.metrics[metric]) - float(baseline.metrics[metric]) for item in rows],
                figures / f"{task}_{metric}_delta.png",
            )
    for item in results:
        if item.pr_recall and item.pr_precision:
            write_pr_curve(
                item.pr_recall,
                item.pr_precision,
                figures / "pr" / f"{item.task}_{item.subset}_{item.side_px}.png",
                title=f"{item.task} {item.subset} {item.side_px}px",
            )
    for item in results:
        epochs = [row.epoch for row in item.history]
        train_loss = [row.train_loss for row in item.history]
        test_loss = [row.test_loss for row in item.history]
        if any(value is None for value in train_loss) or any(value is None for value in test_loss):
            raise ValueError(f"{item.task} {item.subset} is missing a loss")
        write_loss_curve(
            epochs,
            [float(value) for value in train_loss if value is not None],
            [row.val_loss for row in item.history],
            [float(value) for value in test_loss if value is not None],
            figures / "losses" / f"{item.task}_{item.subset}_{item.side_px}.png",
            selected_epoch=item.best_epoch + 1,
        )


def _load_saved(path: Path) -> tuple[CellResult, ...]:
    """Read a review JSON file into cell results.

    Args:
        path: ``results.json`` path. A missing file yields no cells.

    Returns:
        tuple[CellResult, ...]: Saved cells. Precision-recall curves are empty.
    """
    if not path.is_file():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    loaded: list[CellResult] = []
    for item in payload:
        history = tuple(
            EpochLoss(
                epoch=int(row["epoch"]),
                train_loss=None if row["train_loss"] is None else float(row["train_loss"]),
                val_loss=float(row["val_loss"]),
                test_loss=None if row["test_loss"] is None else float(row["test_loss"]),
            )
            for row in item["history"]
        )
        loaded.append(
            CellResult(
                task=str(item["task"]),
                subset=str(item["subset"]),
                side_px=int(item["side_px"]),
                best_epoch=int(item["best_epoch"]),
                metrics={str(key): float(value) for key, value in item["metrics"].items()},
                history=history,
            )
        )
    return tuple(loaded)


def _parser() -> argparse.ArgumentParser:
    """Return the native-sweep parser."""
    parser = argparse.ArgumentParser(prog="original-dataset-analysis train-native")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only", action="append", default=[], help="task:subset to train")
    parser.add_argument("--preview", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--axis", choices=("native", "gsd"), default="native")
    return parser


def train_native(argv: Sequence[str]) -> int:
    """Train native cells and write the review bundle.

    Args:
        argv: Arguments after ``train-native``.

    Returns:
        int: Zero after the review files are written.

    Raises:
        ValueError: If ``--device cuda`` is requested and CUDA is unavailable,
            or ``--only`` does not name a native cell.
    """
    args = _parser().parse_args(list(argv))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    device = torch.device(args.device)
    index = build_index(args.images, args.labels)
    if (args.cache / "meta.json").is_file():
        cache = open_cache(args.cache)
    else:
        cache = build_cache(args.images, index.tiles, args.cache)
    order = verify_band_order(coerce_descriptions(cache.descriptions))
    mask_path = args.cache / "masks.dat"
    mask_count = len(index.tiles) * 120 * 120
    if mask_path.is_file() and mask_path.stat().st_size == mask_count:
        cached_masks: np.ndarray = np.memmap(
            mask_path, dtype=np.uint8, mode="r", shape=(len(index.tiles), 120, 120)
        )
    else:
        print("rasterizing masks", flush=True)
        cached_masks = build_mask_cache(index.tiles, mask_path)
    mask_rows = {tile.stem: row for row, tile in enumerate(index.tiles)}
    mask_by_side: dict[int, np.ndarray] = {120: cached_masks}
    planned = gsd_cells(order) if args.axis == "gsd" else native_cells(order)
    if args.axis == "gsd":
        planned = tuple(cell for cell in planned if cell.side_px != 120)
    if args.only:
        wanted = {tuple(item.split(":", 1)) for item in args.only}
        planned = tuple(cell for cell in planned if (cell.task, cell.subset) in wanted)
        if len(planned) != len(wanted):
            raise ValueError(f"unknown cells in {args.only}")
    if args.preview:
        write_mask_previews(cache, index.tiles, args.out / "previews", args.preview)
    config = TrainConfig(epochs=args.epochs, seed=0)
    split = assign_location_splits(index, SplitRecipe(seed=config.seed))
    prevalence = {
        name: sum(tile.positive for tile in _tiles(index, stems)) / float(len(stems))
        for name, stems in split.stems.items()
        if stems
    }
    prior = _load_saved(args.out / "results.json")
    done = {(item.task, item.subset, item.side_px) for item in prior}
    planned = tuple(cell for cell in planned if (cell.task, cell.subset, cell.side_px) not in done)
    if prior:
        write_review(order, prior, args.out, prevalence=prevalence)

    def _publish(partial: tuple[CellResult, ...]) -> None:
        write_review(order, prior + partial, args.out, prevalence=prevalence)
        print(f"saved {len(prior) + len(partial)} cells", flush=True)

    results: tuple[CellResult, ...] = ()
    if args.axis == "gsd":
        cache.close()
        accumulated = list(prior)

        def _publish_gsd(partial: tuple[CellResult, ...]) -> None:
            write_review(order, tuple(accumulated) + partial, args.out, prevalence=prevalence)
            print(f"saved {len(accumulated) + len(partial)} cells", flush=True)

        for side in (80, 60, 40, 30):
            side_cells = tuple(cell for cell in planned if cell.side_px == side)
            if not side_cells:
                continue
            pack_dir = args.cache.parent / "packs" / str(side)
            if not (pack_dir / "meta.json").is_file():
                print(f"preparing pack {side}", flush=True)
                native = open_cache(args.cache)
                mask_file = args.cache / f"masks_{side}.dat"
                prepare_side(
                    native,
                    index.tiles,
                    side,
                    pack_dir,
                    mask_source=mask_file if mask_file.is_file() else None,
                )
                native.close()
            pack = open_side_pack(pack_dir)
            batch = run_native(
                index,
                cache,
                order,
                side_cells,
                config,
                device,
                pack=pack,
                on_result=_publish_gsd,
            )
            accumulated.extend(batch)
            pack.close()
        results = tuple(accumulated[len(prior) :])
    else:
        results = run_native(
            index,
            cache,
            order,
            planned,
            config,
            device,
            cached_masks=cached_masks,
            mask_rows=mask_rows,
            mask_by_side=mask_by_side,
            on_result=_publish,
        )
    write_review(order, prior + results, args.out, prevalence=prevalence)
    print(f"wrote {len(results)} cells to {args.out}")
    return 0
