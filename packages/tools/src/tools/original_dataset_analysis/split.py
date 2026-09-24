"""Location-grouped train, validation, and test assignment.

Contains:
  - SplitRecipe: seed and site fractions.
  - LocationSplit: the site assignment and the stems in each split.
  - assign_location_splits: one split name per location id.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tools.original_dataset_analysis.index import TileIndex


@dataclass(frozen=True, slots=True)
class SplitRecipe:
    """Seeded fractions of sites.

    Attributes:
        seed: Shuffle seed.
        train_fraction: Share of sites in train.
        val_fraction: Share of sites in validation.
        test_fraction: Share of sites in test.
    """

    seed: int
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15


@dataclass(frozen=True, slots=True)
class LocationSplit:
    """Every tile of a location follows that location.

    Attributes:
        location_to_split: Split name for each location id.
        stems: Image stems in each split, archive order.
    """

    location_to_split: dict[str, str]
    stems: dict[str, tuple[str, ...]]


def assign_location_splits(index: TileIndex, recipe: SplitRecipe) -> LocationSplit:
    """Assign each location id to train, val, or test.

    Args:
        index: Corpus index.
        recipe: Seed and fractions. Fractions must be positive and sum to 1.

    Returns:
        LocationSplit: Site assignment and the stems that follow it.

    Raises:
        ValueError: If the recipe is invalid or fewer than three sites exist.
    """
    parts = (recipe.train_fraction, recipe.val_fraction, recipe.test_fraction)
    if any(part <= 0.0 for part in parts):
        raise ValueError("split fractions must be > 0")
    if abs(sum(parts) - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0; got {sum(parts)}")
    locations = tuple(dict.fromkeys(tile.location_id for tile in index.tiles))
    if len(locations) < 3:
        raise ValueError(f"need at least 3 locations to split; got {len(locations)}")
    generator = torch.Generator()
    generator.manual_seed(recipe.seed)
    order = [locations[int(value)] for value in torch.randperm(len(locations), generator=generator)]
    n_val = max(1, int(round(recipe.val_fraction * len(order))))
    n_test = max(1, int(round(recipe.test_fraction * len(order))))
    if n_val + n_test >= len(order):
        n_val = 1
        n_test = 1
    n_train = len(order) - n_val - n_test
    names = ["train"] * n_train + ["val"] * n_val + ["test"] * (len(order) - n_train - n_val)
    location_to_split = dict(zip(order, names, strict=True))
    grouped: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for tile in index.tiles:
        grouped[location_to_split[tile.location_id]].append(tile.stem)
    stems = {name: tuple(values) for name, values in grouped.items()}
    return LocationSplit(location_to_split=location_to_split, stems=stems)
