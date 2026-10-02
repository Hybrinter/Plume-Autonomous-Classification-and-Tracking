"""Tests for the finished-dataset training loop, provenance, and evaluation."""

from collections.abc import Iterator
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
from tools.ml_models.dataset.manifest import load_manifest
from tools.ml_models.dataset.raw import BinSpec, GsdPair, RawTile, RawTileRef
from tools.ml_models.dataset.spec import BuildSpec
from tools.ml_models.dataset.split import SplitRecipe
from tools.ml_models.dataset.store import read_rows
from tools.ml_models.train import loop as loop_module
from tools.ml_models.train.config import TrainConfig
from tools.ml_models.train.evaluate import evaluate
from tools.ml_models.train.loop import train
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


def _two_datasets(tmp_path: Path) -> tuple[Path, Path]:
    """Two finished datasets with different GSD extents and tile shapes."""
    first = _build(tmp_path / "a", _groups("a", GsdPair(5.0, 10.0), (8, 16)), _spec())
    second = _build(
        tmp_path / "b",
        _groups("b", GsdPair(2.0, 4.0), (40, 20)),
        _spec(),
    )
    return first, second


def test_tiny_train_over_two_datasets(tmp_path: Path) -> None:
    """A short CPU run trains over two mixed-extent datasets and writes a full
    checkpoint, config, summary, and history."""
    first, second = _two_datasets(tmp_path)
    cfg = TrainConfig(
        kind="segmentor",
        arch="dilatenet_w8_d2",
        datasets=(str(first), str(second)),
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
    assert len(checkpoint["dataset_hash"]) == 64
    assert checkpoint["dataset_weights"] == [1.0, 1.0]

    provenance = checkpoint["provenance"]
    assert provenance["train_samples"] > 0
    assert provenance["gsd_reference_m"] > 0
    assert [entry["dataset_hash"] for entry in provenance["datasets"]] == [
        load_manifest(first / "dataset.json").dataset_hash,
        load_manifest(second / "dataset.json").dataset_hash,
    ]
    # Actual training GSD bounds cover both datasets' train rows.
    assert provenance["gsd_min_m"] == pytest.approx([2.0, 4.0])
    assert provenance["gsd_max_m"] == pytest.approx([5.0, 10.0])

    # The checkpoint forwards through the registry with an (image, gsd) call.
    model = build_model("segmentor", "dilatenet_w8_d2", cast(int, provenance["in_channels"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    shard_dir = next((first / "segmentor" / "val").iterdir())
    dataset = ShardDataset(shard_dir, provenance["gsd_reference_m"], "segmentor", channels=3)
    images, gsd, targets = next(iter(DataLoader(dataset, batch_size=2)))
    with torch.no_grad():
        output: torch.Tensor = model(images, gsd)
    assert output.shape == targets.shape


def test_run_directory_rejects_existing(tmp_path: Path) -> None:
    """An existing run directory is refused; overwrite never deletes."""
    first, second = _two_datasets(tmp_path)
    cfg = TrainConfig(
        kind="classifier",
        arch="pactnet_w8_d2",
        datasets=(str(first), str(second)),
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
        datasets=(str(first), str(second)),
        epochs=1,
        device="cpu",
        run_dir=str(tmp_path / "runs"),
        run_id="fixed",
        overwrite=True,
    )
    assert isinstance(train(reserved), Err)


def test_checkpoint_path_copy_never_overwrites(tmp_path: Path) -> None:
    """``checkpoint_path`` copies last.pt and refuses an existing file."""
    first, second = _two_datasets(tmp_path)
    target = tmp_path / "out" / "model.pt"
    cfg = TrainConfig(
        kind="classifier",
        arch="pactnet_w8_d2",
        datasets=(str(first), str(second)),
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
    first, second = _two_datasets(tmp_path)
    state_dicts = []
    for name in ("r1", "r2"):
        cfg = TrainConfig(
            kind="classifier",
            arch="pactnet_w8_d2",
            datasets=(str(first), str(second)),
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
    first, second = _two_datasets(tmp_path)
    cfg = TrainConfig(
        kind="classifier",
        arch="pactnet_w8_d2",
        datasets=(str(first), str(second)),
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
    first, second = _two_datasets(tmp_path)

    def _flat_report(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "split": "val",
            "aggregation": "dataset_weighted_macro",
            "datasets": [],
            "combined": {"f1": 0.5, "n": 1.0},
        }

    monkeypatch.setattr(loop_module, "evaluate", _flat_report)
    cfg = TrainConfig(
        kind="classifier",
        arch="pactnet_w8_d2",
        datasets=(str(first), str(second)),
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
    first, second = _two_datasets(tmp_path)
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
        datasets=(str(first), str(second)),
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
    first, second = _two_datasets(tmp_path)
    classifier_only = _build(
        tmp_path / "c",
        _groups("c", GsdPair(5.0, 10.0)),
        BuildSpec(split=SplitRecipe(seed=0), augment=_ID_ONLY, tasks=("classifier",)),
    )
    cfg = TrainConfig(
        kind="segmentor",
        arch="dilatenet_w8_d2",
        datasets=(str(first), str(classifier_only)),
        epochs=1,
        device="cpu",
        run_dir=str(tmp_path / "runs"),
    )
    assert isinstance(train(cfg), Err)


def test_mismatched_bands_rejected(tmp_path: Path) -> None:
    """Datasets disagreeing on band order cannot train together."""
    first, _ = _two_datasets(tmp_path)
    other = _build(
        tmp_path / "odd",
        _groups("o", GsdPair(5.0, 10.0)),
        BuildSpec(
            split=SplitRecipe(seed=0),
            augment=_ID_ONLY,
            input_bands=("A", "B", "C"),
        ),
        band_names=("A", "B", "C"),
    )
    cfg = TrainConfig(
        kind="classifier",
        arch="pactnet_w8_d2",
        datasets=(str(first), str(other)),
        epochs=1,
        device="cpu",
        run_dir=str(tmp_path / "runs"),
    )
    assert isinstance(train(cfg), Err)


def test_mismatched_gsd_reference_rejected(tmp_path: Path) -> None:
    """Datasets disagreeing on the GSD reference cannot train together."""
    first, _ = _two_datasets(tmp_path)
    other = _build(
        tmp_path / "odd",
        _groups("o", GsdPair(5.0, 10.0)),
        BuildSpec(split=SplitRecipe(seed=0), augment=_ID_ONLY, gsd_reference_m=9.0),
    )
    cfg = TrainConfig(
        kind="classifier",
        arch="pactnet_w8_d2",
        datasets=(str(first), str(other)),
        epochs=1,
        device="cpu",
        run_dir=str(tmp_path / "runs"),
    )
    assert isinstance(train(cfg), Err)


def test_cross_source_ref_group_leakage_rejected(tmp_path: Path) -> None:
    """One group in train for one dataset and val for another is refused."""
    # Seed 1 puts group "shared" in train; seed 0 puts it in val.
    train_side = _build(
        tmp_path / "a",
        (_tile("shared", "shared", GsdPair(5.0, 10.0)),) + _groups("a", GsdPair(5.0, 10.0)),
        _spec(seed=1),
    )
    val_side = _build(
        tmp_path / "b",
        (_tile("shared", "shared", GsdPair(5.0, 10.0)),) + _groups("a", GsdPair(5.0, 10.0)),
        _spec(seed=0),
    )
    manifests = [
        load_manifest(train_side / "dataset.json"),
        load_manifest(val_side / "dataset.json"),
    ]
    with pytest.raises(ValueError, match="leak"):
        training_provenance([train_side, val_side], manifests, "classifier")


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
    report = evaluate(model, [dest], [manifest], "classifier", "val", 2, (1.0,), "cpu")
    entry = cast(list[dict[str, object]], report["datasets"])[0]
    val_rows = [
        row
        for shard in manifest.shards
        if shard.task == "classifier" and shard.split == "val"
        for row in read_rows(dest / "classifier" / "val" / f"{shard.height}x{shard.width}")
    ]
    per_bin: dict[str, int] = {}
    for row in val_rows:
        per_bin[row.bin_id] = per_bin.get(row.bin_id, 0) + 1
    bins_report = cast(dict[str, dict[str, float]], entry["bins"])
    assert set(bins_report) == set(per_bin)
    for bin_id, count in per_bin.items():
        assert bins_report[bin_id]["n"] == float(count)


def test_evaluate_weighted_macro_combination(tmp_path: Path) -> None:
    """Combined metrics are the explicit dataset-weighted macro average."""
    first, second = _two_datasets(tmp_path)
    manifests = [
        load_manifest(first / "dataset.json"),
        load_manifest(second / "dataset.json"),
    ]
    model = build_model("classifier", "pactnet_w8_d2", 3)
    weights = (1.0, 3.0)
    report = evaluate(model, [first, second], manifests, "classifier", "val", 2, weights, "cpu")
    combined = cast(dict[str, float], report["combined"])
    reports = cast(list[dict[str, object]], report["datasets"])
    metrics = [cast(dict[str, float], entry["metrics"]) for entry in reports]
    for key, value in combined.items():
        if key == "n":
            continue
        expected = sum(weight * item[key] for weight, item in zip(weights, metrics)) / sum(weights)
        assert value == pytest.approx(expected), key


def test_cli_train_echoes_run_and_rejects_bad_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``ml-models train`` prints the run path and maps errors to exit codes."""
    from tools.ml_models.cli import main

    dataset = _build(tmp_path / "ds", _groups("g", GsdPair(5.0, 10.0)), _spec())
    assert main(["train", "--run-dir", str(tmp_path / "runs")]) != 0
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
