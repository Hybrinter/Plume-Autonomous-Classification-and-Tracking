"""Frozen train/val/test split recipe and processed-pack dataset hash.

Split assignment is a seeded permutation of sample indices. Fractions come from
a committed TOML recipe. The recipe applies to sorted paired filenames, so the
split does not need a committed id list.

Contains:
  - SplitRecipe: seed plus train/val/test fractions.
  - SplitIndex: integer index tuples for each split.
  - DatasetMeta: pack identity written next to the npy tensors.
  - load_split_recipe / assign_splits / write_splits / load_splits.
  - compute_dataset_hash / write_dataset_meta / load_dataset_meta.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    """Return the repository root that holds data/manifests."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data" / "manifests").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[5]


DEFAULT_SPLIT_RECIPE = _repo_root() / "data" / "manifests" / "zenodo_4250706_splits.toml"
_PACK_HASH_FILES: tuple[str, ...] = ("images.npy", "masks.npy", "labels.npy", "splits.json")
_SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class SplitRecipe:
    """Seeded fractional split recipe.

    Attributes:
        seed: RNG seed for the index permutation.
        train_fraction: Fraction of samples assigned to train.
        val_fraction: Fraction of samples assigned to val.
        test_fraction: Fraction of samples assigned to test.
    """

    seed: int = 0
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15


@dataclass(frozen=True, slots=True)
class SplitIndex:
    """Integer sample indices for train, val, and test.

    Attributes:
        train: Train indices in permutation order.
        val: Validation indices.
        test: Test indices.
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


@dataclass(frozen=True, slots=True)
class DatasetMeta:
    """Identity sidecar for a processed pack.

    Attributes:
        dataset_hash: SHA-256 over the pack file digests.
        source_doi: Dataset DOI string. Empty when synthetic.
        n: Sample count N.
        height: Spatial height H.
        width: Spatial width W.
        in_channels: Band count C.
    """

    dataset_hash: str
    source_doi: str
    n: int
    height: int
    width: int
    in_channels: int


def load_split_recipe(path: str | Path | None = None) -> SplitRecipe:
    """Parse a split-recipe TOML file.

    Args:
        path: TOML path. None uses ``data/manifests/zenodo_4250706_splits.toml``.

    Returns:
        SplitRecipe: Seed and fractions.

    Raises:
        OSError / tomllib.TOMLDecodeError / KeyError: on a missing or malformed file.
        ValueError: If fractions are not positive or do not sum to 1.
    """
    dest = Path(path) if path is not None else DEFAULT_SPLIT_RECIPE
    data = tomllib.loads(dest.read_text(encoding="utf-8"))
    recipe = SplitRecipe(
        seed=int(data.get("seed", 0)),
        train_fraction=float(data.get("train_fraction", 0.70)),
        val_fraction=float(data.get("val_fraction", 0.15)),
        test_fraction=float(data.get("test_fraction", 0.15)),
    )
    _validate_recipe(recipe)
    return recipe


def _validate_recipe(recipe: SplitRecipe) -> None:
    """Raise ValueError when fractions are invalid."""
    parts = (recipe.train_fraction, recipe.val_fraction, recipe.test_fraction)
    if any(part <= 0.0 for part in parts):
        raise ValueError("split fractions must be > 0")
    total = sum(parts)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0; got {total}")


def assign_splits(n: int, recipe: SplitRecipe) -> SplitIndex:
    """Assign every index in ``range(n)`` to train, val, or test.

    Args:
        n: Sample count. Must be >= 3 so each split can hold one index.
        recipe: Seeded fractions.

    Returns:
        SplitIndex: Disjoint index tuples that cover ``0..n-1``.

    Raises:
        ValueError: If n < 3 or the recipe is invalid.

    Notes:
        The permutation uses ``numpy.random.Generator``. Rounding keeps at least
        one sample in each split when n >= 3. Leftover indices go to train.
    """
    _validate_recipe(recipe)
    if n < 3:
        raise ValueError(f"need at least 3 samples to split; got {n}")
    rng = np.random.default_rng(recipe.seed)
    order = [int(v) for v in rng.permutation(n)]
    n_val = max(1, int(round(recipe.val_fraction * n)))
    n_test = max(1, int(round(recipe.test_fraction * n)))
    if n_val + n_test >= n:
        n_val = 1
        n_test = 1
    n_train = n - n_val - n_test
    train = tuple(order[:n_train])
    val = tuple(order[n_train : n_train + n_val])
    test = tuple(order[n_train + n_val :])
    return SplitIndex(train=train, val=val, test=test)


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
        OSError / json.JSONDecodeError / KeyError: on a missing or malformed file.
        ValueError: If a split name is missing or indices overlap.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    index = SplitIndex(
        train=tuple(int(v) for v in data["train"]),
        val=tuple(int(v) for v in data["val"]),
        test=tuple(int(v) for v in data["test"]),
    )
    seen: set[int] = set()
    for name in _SPLIT_NAMES:
        for item in index.for_name(name):
            if item in seen:
                raise ValueError(f"duplicate index {item} in splits")
            seen.add(item)
    return index


def compute_dataset_hash(pack_dir: str | Path) -> str:
    """Return a SHA-256 over the pack file digests.

    Args:
        pack_dir: Directory with ``images.npy``, ``masks.npy``, ``labels.npy``,
            and ``splits.json``.

    Returns:
        str: Lowercase hex digest.

    Raises:
        FileNotFoundError: If a required pack file is missing.

    Notes:
        The hash is the SHA-256 of ``name:file_sha256`` lines. It does not hash
        the full concatenated bytes, so it stays cheap on a large corpus.
    """
    root = Path(pack_dir)
    hasher = hashlib.sha256()
    for name in _PACK_HASH_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"missing pack file {path}")
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hasher.update(f"{name}:{file_digest}\n".encode())
    return hasher.hexdigest()


def write_dataset_meta(path: str | Path, meta: DatasetMeta) -> None:
    """Write DatasetMeta as JSON.

    Args:
        path: Destination ``dataset.json``.
        meta: Pack identity.

    Returns:
        None.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_hash": meta.dataset_hash,
        "source_doi": meta.source_doi,
        "n": meta.n,
        "height": meta.height,
        "width": meta.width,
        "in_channels": meta.in_channels,
    }
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_dataset_meta(path: str | Path) -> DatasetMeta:
    """Parse ``dataset.json`` into DatasetMeta.

    Args:
        path: JSON path.

    Returns:
        DatasetMeta: Loaded identity.

    Raises:
        OSError / json.JSONDecodeError / KeyError: on a missing or malformed file.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return DatasetMeta(
        dataset_hash=str(data["dataset_hash"]),
        source_doi=str(data["source_doi"]),
        n=int(data["n"]),
        height=int(data["height"]),
        width=int(data["width"]),
        in_channels=int(data["in_channels"]),
    )
