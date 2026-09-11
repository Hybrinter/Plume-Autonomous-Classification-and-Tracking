"""Tests for frozen train/val/test splits and dataset hashing."""

from pathlib import Path

import numpy as np
import pytest
from tools.inference.split import (
    DatasetMeta,
    SplitIndex,
    SplitRecipe,
    assign_splits,
    compute_dataset_hash,
    load_dataset_meta,
    load_split_recipe,
    load_splits,
    write_dataset_meta,
    write_splits,
)


def test_load_split_recipe_defaults() -> None:
    """The committed recipe is 70/15/15 with seed 0."""
    recipe = load_split_recipe()
    assert recipe.seed == 0
    assert recipe.train_fraction == pytest.approx(0.70)
    assert recipe.val_fraction == pytest.approx(0.15)
    assert recipe.test_fraction == pytest.approx(0.15)


def test_assign_splits_covers_all_indices() -> None:
    """assign_splits permutes 0..n-1 into disjoint nonempty splits."""
    index = assign_splits(10, SplitRecipe(seed=0))
    combined = set(index.train) | set(index.val) | set(index.test)
    assert combined == set(range(10))
    assert not set(index.train) & set(index.val)
    assert not set(index.train) & set(index.test)
    assert not set(index.val) & set(index.test)
    assert len(index.train) >= 1
    assert len(index.val) >= 1
    assert len(index.test) >= 1


def test_assign_splits_is_seed_deterministic() -> None:
    """The same seed yields the same index tuples."""
    recipe = SplitRecipe(seed=3)
    first = assign_splits(8, recipe)
    second = assign_splits(8, recipe)
    assert first == second


def test_assign_splits_rejects_small_n() -> None:
    """Fewer than 3 samples cannot fill three splits."""
    with pytest.raises(ValueError, match="at least 3"):
        assign_splits(2, SplitRecipe())


def test_load_split_recipe_rejects_bad_fractions(tmp_path: Path) -> None:
    """Fractions that do not sum to 1 raise ValueError."""
    path = tmp_path / "bad.toml"
    path.write_text("seed = 0\ntrain_fraction = 0.5\nval_fraction = 0.5\ntest_fraction = 0.5\n")
    with pytest.raises(ValueError, match="sum to 1"):
        load_split_recipe(path)


def test_splits_json_roundtrip(tmp_path: Path) -> None:
    """write_splits / load_splits preserve index tuples."""
    index = SplitIndex(train=(0, 1), val=(2,), test=(3,))
    path = tmp_path / "splits.json"
    write_splits(path, index)
    loaded = load_splits(path)
    assert loaded == index


def test_load_splits_rejects_duplicates(tmp_path: Path) -> None:
    """Overlapping split indices raise ValueError."""
    path = tmp_path / "splits.json"
    path.write_text('{"train": [0, 1], "val": [1], "test": [2]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_splits(path)


def test_dataset_hash_changes_when_a_file_changes(tmp_path: Path) -> None:
    """compute_dataset_hash is sensitive to pack file bytes."""
    images = np.zeros((3, 4, 2, 2), dtype=np.float32)
    masks = np.zeros((3, 1, 2, 2), dtype=np.float32)
    labels = np.zeros((3, 1), dtype=np.float32)
    np.save(tmp_path / "images.npy", images)
    np.save(tmp_path / "masks.npy", masks)
    np.save(tmp_path / "labels.npy", labels)
    write_splits(tmp_path / "splits.json", assign_splits(3, SplitRecipe(seed=0)))
    first = compute_dataset_hash(tmp_path)
    labels[0, 0] = 1.0
    np.save(tmp_path / "labels.npy", labels)
    second = compute_dataset_hash(tmp_path)
    assert first != second


def test_dataset_meta_roundtrip(tmp_path: Path) -> None:
    """write_dataset_meta / load_dataset_meta preserve fields."""
    meta = DatasetMeta(
        dataset_hash="abc",
        source_doi="10.5281/zenodo.4250706",
        n=3,
        height=8,
        width=8,
        in_channels=4,
    )
    path = tmp_path / "dataset.json"
    write_dataset_meta(path, meta)
    assert load_dataset_meta(path) == meta
