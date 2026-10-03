"""Tests for the finished-dataset training loop, provenance, and evaluation."""

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from flight.libs.types import Err, Ok
from tools.ml_models.arch.film import CONDITIONING_ID
from tools.ml_models.arch.registry import build as build_model
from tools.ml_models.dataset.augment import AugmentRecipe
from tools.ml_models.dataset.build import build_dataset
from tools.ml_models.dataset.loader import ShardDataset
from tools.ml_models.dataset.manifest import DatasetManifest, load_manifest
from tools.ml_models.dataset.raw import BinSpec, GsdPair, RawTile, RawTileRef
from tools.ml_models.dataset.spec import BuildSpec
from tools.ml_models.dataset.split import SplitRecipe
from tools.ml_models.dataset.store import read_gsd, read_images, read_labels, read_masks, read_rows
from tools.ml_models.train import loop as loop_module
from tools.ml_models.train.config import TrainConfig
from tools.ml_models.train.evaluate import evaluate
from tools.ml_models.train.loop import train
from tools.ml_models.train.metrics import classifier_metrics, compute_dice, compute_iou
from tools.ml_models.train.provenance import training_provenance
from torch import nn
from torch.utils.data import DataLoader


class MemorySource:
    """In-memory raw source for training tests."""

    def __init__(
        self,
        tiles: tuple[RawTile, ...],
        *,
        band_names: tuple[str, ...] = ("BLUE", "GREEN", "RED"),
        source_ref: str = "test",
        bins: tuple[BinSpec, ...] = (),
    ) -> None:
        self.name = "memory"
        self.band_names = band_names
        self.domain = "unit"
        self.source_ref = source_ref
        self.bins = bins
        self._tiles = tiles

    def index(self) -> tuple[RawTileRef, ...]:
        """Return refs in stream order."""
        return tuple(tile.ref for tile in self._tiles)

    def iter_tiles(self) -> Iterator[RawTile]:
        """Yield the tiles once."""
        yield from self._tiles


def _tile(
    tile_id: str,
    group_id: str,
    gsd: GsdPair,
    shape: tuple[int, int] = (8, 16),
    *,
    label: float = 1.0,
    has_mask: bool = True,
    bin_id: str = "",
) -> RawTile:
    """Build one small raw tile."""
    height, width = shape
    image = np.arange(3 * height * width, dtype=np.float32).reshape(3, height, width) % 1024
    image = image / np.float32(1024.0)
    mask = None
    if has_mask:
        mask = np.zeros((1, height, width), dtype=np.uint8)
        mask[:, : height // 2, :] = 1
    return RawTile(
        ref=RawTileRef(
            tile_id=tile_id,
            group_id=group_id,
            label=label,
            has_mask=has_mask,
            gsd=gsd,
            height=height,
            width=width,
            frame_id=group_id,
            grid_rc=None,
            bin_id=bin_id,
        ),
        image=image,
        mask=mask,
    )


def _groups(
    prefix: str,
    gsd: GsdPair,
    shape: tuple[int, int] = (8, 16),
    *,
    bins: tuple[str, ...] = (),
) -> tuple[RawTile, ...]:
    """Six groups of tiles at one GSD, optionally duplicated per bin."""
    tiles: list[RawTile] = []
    for index in range(6):
        group = f"{prefix}{index}"
        if bins:
            for bin_id in bins:
                tiles.append(
                    _tile(
                        f"{group}-{bin_id}",
                        group,
                        gsd,
                        shape,
                        label=float(index < 3),
                        bin_id=bin_id,
                    )
                )
        else:
            tiles.append(_tile(group, group, gsd, shape, label=float(index < 3)))
    return tuple(tiles)


_ID_ONLY = AugmentRecipe(elements=("id",))


def _spec(seed: int = 0) -> BuildSpec:
    """Small deterministic build: identity-only train augmentation."""
    return BuildSpec(split=SplitRecipe(seed=seed), augment=_ID_ONLY)


def _build(
    dest: Path,
    tiles: tuple[RawTile, ...],
    spec: BuildSpec,
    *,
    band_names: tuple[str, ...] = ("BLUE", "GREEN", "RED"),
    source_ref: str = "test",
    bins: tuple[BinSpec, ...] = (),
) -> Path:
    """Build a finished dataset and return its directory."""
    build_dataset(
        MemorySource(
            tiles,
            band_names=band_names,
            source_ref=source_ref,
            bins=bins,
        ),
        dest,
        spec,
    )
    return dest


def _dataset(tmp_path: Path) -> Path:
    """One finished dataset mixing two GSD extents and tile shapes."""
    return _build(
        tmp_path / "ds",
        _groups("a", GsdPair(5.0, 10.0), (8, 16)) + _groups("b", GsdPair(2.0, 4.0), (40, 20)),
        _spec(),
    )


def test_tiny_train_on_one_dataset(tmp_path: Path) -> None:
    """A short CPU run writes a full checkpoint, config, summary, and history."""
    dataset = _dataset(tmp_path)
    cfg = TrainConfig(
        kind="segmentor",
        arch="dilatenet_w8_d2",
        dataset=str(dataset),
        epochs=2,
        batch_size=2,
        device="cpu",
        run_dir=str(tmp_path / "runs"),
    )
    result = train(cfg)
    assert isinstance(result, Ok)
    run = result.value
    names = {path.name for path in run.iterdir()}
    assert {"checkpoints", "config.toml", "summary.json", "history.jsonl"} <= names
    assert (run / "checkpoints" / "best.pt").is_file()
    assert (run / "checkpoints" / "last.pt").is_file()

    checkpoint = torch.load(run / "checkpoints" / "last.pt", weights_only=False)
    assert checkpoint["kind"] == "segmentor"
    assert checkpoint["arch"] == "dilatenet_w8_d2"
    assert checkpoint["conditioning"] == CONDITIONING_ID
    assert checkpoint["config"]["epochs"] == 2
    manifest = load_manifest(dataset / "dataset.json")
    assert checkpoint["dataset_hash"] == manifest.dataset_hash
    assert "dataset_weights" not in checkpoint
    assert "input_height_px" not in checkpoint
    assert "input_width_px" not in checkpoint

    provenance = checkpoint["provenance"]
    assert provenance["train_samples"] > 0
    assert provenance["gsd_reference_m"] > 0
    entry = cast(dict[str, object], provenance["dataset"])
    assert entry["path"] == str(dataset)
    assert entry["dataset_hash"] == manifest.dataset_hash
    assert provenance["spatial_shapes"] == [(8, 16), (40, 20)]
    # Measured training GSD bounds cover the dataset's train rows.
    assert provenance["gsd_min_m"] == pytest.approx([2.0, 4.0])
    assert provenance["gsd_max_m"] == pytest.approx([5.0, 10.0])

    # The checkpoint forwards through the registry with an (image, gsd) call.
    model = build_model("segmentor", "dilatenet_w8_d2", cast(int, provenance["in_channels"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    shard_dir = next((dataset / "segmentor" / "val").iterdir())
    shard = ShardDataset(shard_dir, provenance["gsd_reference_m"], "segmentor", channels=3)
    images, gsd, targets = next(iter(DataLoader(shard, batch_size=2)))
    with torch.no_grad():
        output: torch.Tensor = model(images, gsd)
    assert output.shape == targets.shape


def test_run_directory_rejects_existing(tmp_path: Path) -> None:
    """An existing run directory is refused; overwrite never deletes."""
    dataset = _dataset(tmp_path)
    cfg = TrainConfig(
        kind="classifier",
        arch="pactnet_w8_d2",
        dataset=str(dataset),
        epochs=1,
        device="cpu",
        run_dir=str(tmp_path / "runs"),
        run_id="fixed",
    )
    assert isinstance(train(cfg), Ok)
    again = train(cfg)
    assert isinstance(again, Err)
    reserved = TrainConfig(
        kind="classifier",
        arch="pactnet_w8_d2",
        dataset=str(dataset),
        epochs=1,
        device="cpu",
        run_dir=str(tmp_path / "runs"),
        run_id="fixed",
        overwrite=True,
    )
    assert isinstance(train(reserved), Err)


def test_checkpoint_path_copy_never_overwrites(tmp_path: Path) -> None:
    """``checkpoint_path`` copies last.pt and refuses an existing file."""
    dataset = _dataset(tmp_path)
    target = tmp_path / "out" / "model.pt"
    cfg = TrainConfig(
        kind="classifier",
        arch="pactnet_w8_d2",
        dataset=str(dataset),
        epochs=1,
        device="cpu",
        run_dir=str(tmp_path / "runs"),
        checkpoint_path=str(target),
    )
    assert isinstance(train(cfg), Ok)
    assert target.is_file()
    assert isinstance(train(cfg), Err)


def test_seed_reproducibility(tmp_path: Path) -> None:
    """Two runs with one seed produce identical weights."""
    dataset = _dataset(tmp_path)
    state_dicts = []
    for name in ("r1", "r2"):
        cfg = TrainConfig(
            kind="classifier",
            arch="pactnet_w8_d2",
            dataset=str(dataset),
            epochs=1,
            batch_size=2,
            seed=7,
            device="cpu",
            run_dir=str(tmp_path / "runs"),
            run_id=name,
        )
        result = train(cfg)
        assert isinstance(result, Ok)
        state_dicts.append(
            torch.load(result.value / "checkpoints" / "last.pt", weights_only=False)["state_dict"]
        )
    for key, tensor in state_dicts[0].items():
        assert torch.equal(tensor, state_dicts[1][key]), key


def test_max_steps_stops_run(tmp_path: Path) -> None:
    """``max_steps`` halts after the current evaluation and checkpoint."""
    cfg = TrainConfig(
        kind="classifier",
        arch="pactnet_w8_d2",
        dataset=str(_dataset(tmp_path)),
        epochs=4,
        batch_size=2,
        device="cpu",
        run_dir=str(tmp_path / "runs"),
        max_steps=1,
    )
    result = train(cfg)
    assert isinstance(result, Ok)
    checkpoint = torch.load(result.value / "checkpoints" / "last.pt", weights_only=False)
    assert checkpoint["epoch"] == 1
    assert (result.value / "checkpoints" / "best.pt").is_file()


def test_early_stopping_patience(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Evaluations without improvement stop the run before ``epochs``."""

    def _flat_report(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "split": "val",
            "metrics": {"f1": 0.5, "n": 1.0},
        }

    monkeypatch.setattr(loop_module, "evaluate", _flat_report)
    cfg = TrainConfig(
        kind="classifier",
        arch="pactnet_w8_d2",
        dataset=str(_dataset(tmp_path)),
        epochs=5,
        batch_size=2,
        patience=1,
        device="cpu",
        run_dir=str(tmp_path / "runs"),
    )
    result = train(cfg)
    assert isinstance(result, Ok)
    checkpoint = torch.load(result.value / "checkpoints" / "last.pt", weights_only=False)
    assert checkpoint["epoch"] == 2
    history = (result.value / "history.jsonl").read_text().strip().splitlines()
    assert len(history) == 2


def test_nan_loss_errors_before_optimizer_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-finite loss aborts the run before any gradient is applied."""
    dataset = _dataset(tmp_path)
    stepped = False
    real_step = torch.optim.SGD.step

    def _spy_step(self: torch.optim.SGD, *args: object, **kwargs: object) -> object:
        nonlocal stepped
        stepped = True
        return real_step(self, *args, **kwargs)

    class _NanLoss(nn.Module):
        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return torch.full((), float("nan"))

    monkeypatch.setattr(torch.optim.SGD, "step", _spy_step)
    monkeypatch.setattr(loop_module, "build_loss", lambda *a, **k: _NanLoss())
    cfg = TrainConfig(
        kind="classifier",
        arch="pactnet_w8_d2",
        dataset=str(dataset),
        epochs=1,
        batch_size=2,
        device="cpu",
        run_dir=str(tmp_path / "runs"),
        run_id="nan",
    )
    result = train(cfg)
    assert isinstance(result, Err)
    assert not stepped
    run = tmp_path / "runs" / "nan"
    assert not (run / "checkpoints" / "best.pt").exists()
    assert not (run / "checkpoints" / "last.pt").exists()


def test_missing_task_split_rejected(tmp_path: Path) -> None:
    """A dataset with no segmentor rows cannot train a segmentor."""
    classifier_only = _build(
        tmp_path / "c",
        _groups("c", GsdPair(5.0, 10.0)),
        BuildSpec(split=SplitRecipe(seed=0), augment=_ID_ONLY, tasks=("classifier",)),
    )
    cfg = TrainConfig(
        kind="segmentor",
        arch="dilatenet_w8_d2",
        dataset=str(classifier_only),
        epochs=1,
        device="cpu",
        run_dir=str(tmp_path / "runs"),
    )
    assert isinstance(train(cfg), Err)


def _rewrite_rows(shard_dir: Path, mutate: Callable[[int, dict[str, object]], None]) -> None:
    """Apply ``mutate(index, record)`` to each rows.jsonl record in place."""
    path = shard_dir / "rows.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    for index, record in enumerate(records):
        mutate(index, record)
    path.write_text(
        "\n".join(json.dumps(record, separators=(",", ":")) for record in records) + "\n"
    )


def _train_shard(dataset: Path, manifest: DatasetManifest, task: str) -> Path:
    """The shard dir of the first manifest-listed train shard for ``task``."""
    shard = next(
        entry for entry in manifest.shards if entry.task == task and entry.split == "train"
    )
    return dataset / task / "train" / f"{shard.height}x{shard.width}"


def test_group_leakage_across_splits_rejected(tmp_path: Path) -> None:
    """A group reused across train and val shards of one dataset is refused."""
    dataset = _build(tmp_path / "ds", _groups("g", GsdPair(5.0, 10.0)), _spec())
    manifest = load_manifest(dataset / "dataset.json")
    train_dir = _train_shard(dataset, manifest, "classifier")
    train_groups = {row.group_id for row in read_rows(train_dir)}
    val_shard = next(
        entry for entry in manifest.shards if entry.task == "classifier" and entry.split == "val"
    )
    val_dir = dataset / "classifier" / "val" / f"{val_shard.height}x{val_shard.width}"
    _rewrite_rows(
        val_dir,
        lambda index, record: (
            record.update(group_id=sorted(train_groups)[0]) if index == 0 else None
        ),
    )
    with pytest.raises(ValueError, match="leak"):
        training_provenance(dataset, manifest, "classifier")


def test_group_leakage_across_tasks_rejected(tmp_path: Path) -> None:
    """A group in classifier train but segmentor val is refused."""
    dataset = _build(tmp_path / "ds", _groups("g", GsdPair(5.0, 10.0)), _spec())
    manifest = load_manifest(dataset / "dataset.json")
    train_dir = _train_shard(dataset, manifest, "classifier")
    train_groups = {row.group_id for row in read_rows(train_dir)}
    val_shard = next(
        entry for entry in manifest.shards if entry.task == "segmentor" and entry.split == "val"
    )
    val_dir = dataset / "segmentor" / "val" / f"{val_shard.height}x{val_shard.width}"
    _rewrite_rows(
        val_dir,
        lambda index, record: (
            record.update(group_id=sorted(train_groups)[0]) if index == 0 else None
        ),
    )
    with pytest.raises(ValueError, match="leak"):
        training_provenance(dataset, manifest, "classifier")


def test_evaluate_bins_cover_each_val_row_once(tmp_path: Path) -> None:
    """Every val row lands in exactly one named-bin bucket."""
    dest = _build(
        tmp_path / "ds",
        _groups("g", GsdPair(5.0, 10.0), bins=("low", "high")),
        _spec(),
        bins=(BinSpec("low", 5.0, 5.0), BinSpec("high", 10.0, 10.0)),
    )
    manifest = load_manifest(dest / "dataset.json")
    model = build_model("classifier", "pactnet_w8_d2", 3)
    report = evaluate(model, dest, manifest, "classifier", "val", 2, "cpu")
    assert report["dataset"] == str(dest)
    assert report["dataset_hash"] == manifest.dataset_hash
    assert report["source"] == manifest.source
    val_rows = [
        row
        for shard in manifest.shards
        if shard.task == "classifier" and shard.split == "val"
        for row in read_rows(dest / "classifier" / "val" / f"{shard.height}x{shard.width}")
    ]
    per_bin: dict[str, int] = {}
    for row in val_rows:
        per_bin[row.bin_id] = per_bin.get(row.bin_id, 0) + 1
    bins_report = cast(dict[str, dict[str, float]], report["bins"])
    assert set(bins_report) == set(per_bin)
    for bin_id, count in per_bin.items():
        assert bins_report[bin_id]["n"] == float(count)
    metrics = cast(dict[str, float], report["metrics"])
    assert sum(item["n"] for item in bins_report.values()) == pytest.approx(metrics["n"])


def test_cli_train_echoes_run_and_rejects_bad_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``ml-models train`` prints the run path and maps errors to exit codes."""
    from tools.ml_models.cli import main

    dataset = _build(tmp_path / "ds", _groups("g", GsdPair(5.0, 10.0)), _spec())
    assert main(["train", "--run-dir", str(tmp_path / "runs")]) != 0
    assert (
        main(
            [
                "train",
                "--dataset",
                str(dataset),
                "--dataset",
                str(dataset),
                "--run-dir",
                str(tmp_path / "runs"),
            ]
        )
        != 0
    )
    assert main(["train", "--dataset-weight", "1.0", "--dataset", str(dataset)]) != 0
    code = main(
        [
            "train",
            "--kind",
            "classifier",
            "--arch",
            "pactnet_w8_d2",
            "--dataset",
            str(dataset),
            "--run-dir",
            str(tmp_path / "runs"),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--device",
            "cpu",
        ]
    )
    assert code == 0
    assert Path(capsys.readouterr().out.strip()).is_dir()
    assert main(["train", "--config", str(tmp_path / "missing.toml")]) != 0


def test_empty_dataset_fails_before_run_creation(tmp_path: Path) -> None:
    """An empty dataset path errors without creating the run directory."""
    cfg = TrainConfig(
        dataset="",
        epochs=1,
        device="cpu",
        run_dir=str(tmp_path / "runs"),
        run_id="empty",
    )
    assert isinstance(train(cfg), Err)
    assert not (tmp_path / "runs" / "empty").exists()


def test_nominal_rows_do_not_extend_measured_coverage(tmp_path: Path) -> None:
    """A nominal train row still counts but is absent from measured GSD bounds."""
    dataset = _build(tmp_path / "ds", _groups("g", GsdPair(5.0, 10.0)), _spec())
    manifest = load_manifest(dataset / "dataset.json")
    shard_dir = _train_shard(dataset, manifest, "classifier")
    _rewrite_rows(
        shard_dir,
        lambda index, record: record.update(gsd_nominal=True) if index == 0 else None,
    )
    gsd = read_gsd(shard_dir)
    gsd[0] = (1000.0, 1000.0)
    np.save(shard_dir / "gsd.npy", gsd)
    provenance = training_provenance(dataset, manifest, "classifier")
    count = cast(int, provenance["train_samples"])
    assert count == len(read_rows(shard_dir))
    assert provenance["gsd_min_m"] == pytest.approx([5.0, 10.0])
    assert provenance["gsd_max_m"] == pytest.approx([5.0, 10.0])


def test_all_nominal_train_rows_rejected(tmp_path: Path) -> None:
    """A train split with only nominal-GSD rows has no measured coverage."""
    dataset = _build(tmp_path / "ds", _groups("g", GsdPair(5.0, 10.0)), _spec())
    manifest = load_manifest(dataset / "dataset.json")
    shard_dir = _train_shard(dataset, manifest, "classifier")
    _rewrite_rows(shard_dir, lambda index, record: record.update(gsd_nominal=True))
    with pytest.raises(ValueError, match="measured"):
        training_provenance(dataset, manifest, "classifier")


@pytest.mark.parametrize("kind", ["classifier", "segmentor"])
def test_exhaustive_metrics_match_row_formulas(tmp_path: Path, kind: str) -> None:
    """Every variable-shape validation row contributes once to unchanged metrics."""
    from dataclasses import asdict, replace

    class PixelModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen: list[float] = []

        def forward(self, image: torch.Tensor, gsd: torch.Tensor) -> torch.Tensor:
            marker = image[:, 0, 0, 0]
            self.seen.extend(marker.tolist())
            logits = (marker - 0.5) * 4.0
            if kind == "classifier":
                return logits[:, None]
            return logits[:, None, None, None].expand(-1, 1, image.shape[-2], image.shape[-1])

    original = _groups("a", GsdPair(5.0, 10.0), (8, 16), bins=("a",)) + _groups(
        "b", GsdPair(2.0, 4.0), (40, 20), bins=("b",)
    )
    tiles = tuple(
        replace(tile, image=np.full_like(tile.image, np.float32((index + 1) / 13)))
        for index, tile in enumerate(original)
    )
    dataset = _build(tmp_path / "ds", tiles, _spec())
    manifest = load_manifest(dataset / "dataset.json")
    model = PixelModel()
    markers: list[float] = []
    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    overlap: list[tuple[float, float, float, float]] = []
    for shard in manifest.shards:
        if shard.task != kind or shard.split != "val":
            continue
        directory = dataset / kind / "val" / f"{shard.height}x{shard.width}"
        images = read_images(directory)
        labels = read_labels(directory)
        masks = read_masks(directory)
        for index, image in enumerate(images):
            marker = float(image[0, 0, 0])
            markers.append(marker)
            logit = (torch.tensor(marker, dtype=torch.float32) - 0.5) * 4.0
            if kind == "classifier":
                logits.append(logit.reshape(1))
                targets.append(torch.from_numpy(labels[index].copy()))
            else:
                assert masks is not None
                target = torch.from_numpy(masks[index].astype(np.float32))
                prediction = logit.expand_as(target)
                probability = torch.sigmoid(prediction)
                overlap.append(
                    (
                        compute_iou(probability, target, 0.5),
                        compute_dice(probability, target, 0.5),
                        compute_iou(probability, target, 0.55),
                        float(
                            torch.nn.functional.binary_cross_entropy_with_logits(prediction, target)
                        ),
                    )
                )
    report = evaluate(model, dataset, manifest, kind, "val", 3, "cpu")
    assert model.training
    assert model.seen == markers
    metrics = cast(dict[str, float], report["metrics"])
    bins = cast(dict[str, dict[str, float]], report["bins"])
    assert metrics["n"] == len(markers) == sum(values["n"] for values in bins.values())
    if kind == "classifier":
        expected = asdict(classifier_metrics(torch.cat(logits), torch.cat(targets)))
        for name, value in expected.items():
            assert metrics[name] == pytest.approx(value), name
    else:
        for index, name in enumerate(("mean_iou", "mean_dice", "mean_iou_blob_gate", "bce")):
            assert metrics[name] == pytest.approx(
                sum(values[index] for values in overlap) / len(overlap)
            ), name
