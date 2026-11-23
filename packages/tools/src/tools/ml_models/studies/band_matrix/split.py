"""Study location adapter to the shared group-wise dataset split recipe."""

from __future__ import annotations

from dataclasses import dataclass

from tools.ml_models.dataset.sources.zenodo.archive import TileIndex
from tools.ml_models.dataset.split import SplitRecipe, assign_group_splits

__all__ = ["SplitRecipe", "LocationSplit", "assign_location_splits"]


@dataclass(frozen=True, slots=True)
class LocationSplit:
    """Site assignment plus original image stems for each split."""

    location_to_split: dict[str, str]
    stems: dict[str, tuple[str, ...]]


def assign_location_splits(index: TileIndex, recipe: SplitRecipe) -> LocationSplit:
    """Keep all dates from each location in the same split."""
    group_ids = [tile.location_id for tile in index.tiles]
    rows = assign_group_splits(group_ids, recipe)
    location_to_split: dict[str, str] = {}
    stems: dict[str, tuple[str, ...]] = {}
    for name in ("train", "val", "test"):
        indices = rows.for_name(name)
        stems[name] = tuple(index.tiles[row].stem for row in indices)
        for row in indices:
            location_to_split[index.tiles[row].location_id] = name
    return LocationSplit(location_to_split, stems)
