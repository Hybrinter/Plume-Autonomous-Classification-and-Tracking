"""Group-wise train, val, and test indices.

Contains:
  - SplitRecipe: seed plus train/val/test fractions.
  - SplitIndex: integer index tuples for each split.
  - assign_group_splits: one split per group id, rows follow the group.
  - write_splits / load_splits: ``splits.json`` codec.

Fractions apply to unique groups. A numpy Generator shuffles the groups.
Rows are not permuted on their own.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

_SCHEMA = ConfigDict(extra="forbid")
_SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


@pydantic_dataclass(frozen=True, slots=True, config=_SCHEMA)
class SplitRecipe:
    """Seeded fractional split recipe.

    Attributes:
        seed: Seed for ``numpy.random.Generator``.
        train_fraction: Fraction of groups assigned to train.
        val_fraction: Fraction of groups assigned to val.
        test_fraction: Fraction of groups assigned to test.
    """

    seed: int = 0
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15

    @model_validator(mode="after")
    def _fractions_sum_to_one(self) -> Self:
        """Reject non-positive fractions or a set that does not sum to 1."""
        _validate_recipe(self)
        return self


@dataclass(frozen=True, slots=True)
class SplitIndex:
    """Integer sample indices for train, val, and test.

    Attributes:
        train: Train row indices in input order.
        val: Validation row indices in input order.
        test: Test row indices in input order.
    """

    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]

    def for_name(self, name: str) -> tuple[int, ...]:
        """Return the index tuple for ``train``, ``val``, or ``test``.

        Args:
            name: Split name.

        Returns:
            tuple[int, ...]: Indices for that split.

        Raises:
            ValueError: If ``name`` is not a known split.
        """
        if name == "train":
            return self.train
        if name == "val":
            return self.val
        if name == "test":
            return self.test
        raise ValueError(f"unknown split name {name!r}")


def _validate_recipe(recipe: SplitRecipe) -> None:
    """Raise ValueError when fractions are invalid.

    Args:
        recipe: Seeded fractions.

    Returns:
        None.

    Raises:
        ValueError: If a fraction is not finite and positive, or the sum is not 1.
    """
    parts = (recipe.train_fraction, recipe.val_fraction, recipe.test_fraction)
    if any(not math.isfinite(part) or part <= 0.0 for part in parts):
        raise ValueError("split fractions must be finite and > 0")
    total = sum(parts)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0; got {total}")


def assign_group_splits(group_ids: Sequence[str], recipe: SplitRecipe) -> SplitIndex:
    """Assign every row to the split of its group.

    Args:
        group_ids: Group id per row, length N. Unique ids keep first-seen order.
        recipe: Seed and fractions. Fractions apply to groups, not rows.

    Returns:
        SplitIndex: Row indices into ``group_ids``. Each split lists rows in
        input order. The three tuples are disjoint and cover ``0..N-1``.

    Raises:
        ValueError: If the recipe is invalid, a group id is not a string, or
            fewer than 3 unique groups are present.

    Notes:
        Groups are shuffled with ``numpy.random.default_rng(recipe.seed)``,
        which is a ``numpy.random.Generator``. Val and test each receive at
        least one group. When those two counts would consume every group, both
        counts are set to 1 and the leftover groups go to train.
    """
    _validate_recipe(recipe)
    ids = tuple(group_ids)
    if any(not isinstance(group_id, str) for group_id in ids):
        raise ValueError("group ids must be strings")
    unique = tuple(dict.fromkeys(ids))
    if len(unique) < 3:
        raise ValueError(f"need at least 3 groups to split; got {len(unique)}")
    generator = np.random.default_rng(recipe.seed)
    permutation = generator.permutation(len(unique))  # np.ndarray[int64, (G,)]
    order = [unique[int(index)] for index in permutation]
    n_val = max(1, int(round(recipe.val_fraction * len(order))))
    n_test = max(1, int(round(recipe.test_fraction * len(order))))
    if n_val + n_test >= len(order):
        n_val = 1
        n_test = 1
    n_train = len(order) - n_val - n_test
    names = ["train"] * n_train + ["val"] * n_val + ["test"] * (len(order) - n_train - n_val)
    group_to_split = dict(zip(order, names, strict=True))
    grouped: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for row, group_id in enumerate(ids):
        grouped[group_to_split[group_id]].append(row)
    return SplitIndex(
        train=tuple(grouped["train"]),
        val=tuple(grouped["val"]),
        test=tuple(grouped["test"]),
    )


def write_splits(path: str | Path, index: SplitIndex) -> None:
    """Write a SplitIndex as JSON.

    Args:
        path: Destination ``splits.json``.
        index: Split indices.

    Returns:
        None.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {"train": list(index.train), "val": list(index.val), "test": list(index.test)}
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_splits(path: str | Path) -> SplitIndex:
    """Parse ``splits.json`` into a SplitIndex.

    Args:
        path: JSON path.

    Returns:
        SplitIndex: Loaded indices.

    Raises:
        OSError / json.JSONDecodeError: On a missing or malformed file.
        ValueError: If a split name is missing, an index is not an integer,
            an unknown key is present, or indices overlap.
    """
    dest = Path(path)
    data = _read_json_object(dest)
    extra = sorted(set(data) - set(_SPLIT_NAMES))
    if extra:
        raise ValueError(f"unknown split keys {extra}")
    index = SplitIndex(
        train=_index_tuple(data, "train"),
        val=_index_tuple(data, "val"),
        test=_index_tuple(data, "test"),
    )
    seen: set[int] = set()
    for name in _SPLIT_NAMES:
        for item in index.for_name(name):
            if item in seen:
                raise ValueError(f"duplicate index {item} in splits")
            seen.add(item)
    return index


def _read_json_object(path: Path) -> dict[str, object]:
    """Parse a JSON object.

    Args:
        path: File path.

    Returns:
        dict[str, object]: Decoded mapping.

    Raises:
        OSError / json.JSONDecodeError: On a missing or malformed file.
        ValueError: If the root is not a JSON object or a key is not a string.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    decoded: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError(f"{path.name} keys must be strings")
        decoded[key] = value
    return decoded


def _index_tuple(data: Mapping[str, object], name: str) -> tuple[int, ...]:
    """Return one split's indices.

    Args:
        data: Decoded ``splits.json``.
        name: ``train``, ``val``, or ``test``.

    Returns:
        tuple[int, ...]: Indices in file order.

    Raises:
        ValueError: If the key is missing or a value is not an integer.
    """
    if name not in data:
        raise ValueError(f"missing split {name}")
    value = data[name]
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of integers")
    indices: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{name} indices must be integers")
        indices.append(item)
    return tuple(indices)
