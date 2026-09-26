"""Tests for group-wise train, val, and test splits."""

from pathlib import Path

import pytest
from tools.ml_models.data.split import (
    SplitIndex,
    SplitRecipe,
    assign_group_splits,
    load_splits,
    write_splits,
)


def _homes(index: SplitIndex, rows: list[int]) -> list[str]:
    """Return the split names that contain any of ``rows``."""
    found: list[str] = []
    for name in ("train", "val", "test"):
        chosen = index.for_name(name)
        if any(row in chosen for row in rows):
            found.append(name)
    return found


def test_assign_group_splits_keeps_each_group_in_one_split() -> None:
    """Three groups with repeated rows stay whole across train, val, and test."""
    group_ids = ["a", "b", "a", "c", "b", "c", "a"]
    index = assign_group_splits(group_ids, SplitRecipe(seed=0))
    covered = set(index.train) | set(index.val) | set(index.test)
    assert covered == set(range(len(group_ids)))
    assert not set(index.train) & set(index.val)
    assert not set(index.train) & set(index.test)
    assert not set(index.val) & set(index.test)
    assert index.train
    assert index.val
    assert index.test
    assert list(index.train) == sorted(index.train)
    assert list(index.val) == sorted(index.val)
    assert list(index.test) == sorted(index.test)
    for group in ("a", "b", "c"):
        rows = [i for i, group_id in enumerate(group_ids) if group_id == group]
        homes = _homes(index, rows)
        assert len(homes) == 1
        assert all(row in index.for_name(homes[0]) for row in rows)


def test_assign_group_splits_is_seed_deterministic() -> None:
    """The same seed yields the same row indices."""
    group_ids = ["a", "a", "b", "c", "b", "c"]
    recipe = SplitRecipe(seed=4)
    assert assign_group_splits(group_ids, recipe) == assign_group_splits(group_ids, recipe)


def test_assign_group_splits_rejects_fewer_than_three_groups() -> None:
    """Fewer than 3 unique groups raises ValueError."""
    with pytest.raises(ValueError, match="at least 3"):
        assign_group_splits(["a", "b", "a"], SplitRecipe())
    with pytest.raises(ValueError, match="at least 3"):
        assign_group_splits(["only", "only"], SplitRecipe())


def test_split_recipe_rejects_bad_fractions() -> None:
    """Fractions that are not positive or do not sum to 1 raise ValueError."""
    with pytest.raises(ValueError, match="sum to 1"):
        SplitRecipe(train_fraction=0.5, val_fraction=0.5, test_fraction=0.5)
    with pytest.raises(ValueError, match="> 0"):
        SplitRecipe(train_fraction=1.0, val_fraction=0.0, test_fraction=0.0)


def test_split_recipe_defaults() -> None:
    """The default recipe is seed 0 and fractions 0.70/0.15/0.15."""
    recipe = SplitRecipe()
    assert recipe.seed == 0
    assert recipe.train_fraction == pytest.approx(0.70)
    assert recipe.val_fraction == pytest.approx(0.15)
    assert recipe.test_fraction == pytest.approx(0.15)


def test_splits_json_roundtrip(tmp_path: Path) -> None:
    """write_splits / load_splits preserve index tuples."""
    index = SplitIndex(train=(0, 1), val=(2,), test=(3,))
    path = tmp_path / "splits.json"
    write_splits(path, index)
    assert load_splits(path) == index


def test_load_splits_rejects_overlap(tmp_path: Path) -> None:
    """Overlapping split indices raise ValueError."""
    path = tmp_path / "splits.json"
    path.write_text('{"train": [0, 1], "val": [1], "test": [2]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_splits(path)
