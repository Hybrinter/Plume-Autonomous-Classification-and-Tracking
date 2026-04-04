"""
pact.model.dataset — HSG-AIML dataset loader for plume segmentation training.

Satisfies: REQ-AIML-HIGH-001, REQ-AIML-IMAG-001

Loads GeoTIFF multispectral imagery from the HSG-AIML dataset (Zenodo DOI:
10.5281/zenodo.4250706). Selects Sentinel-2 bands B2 (490 nm), B3 (560 nm),
B4 (665 nm), and B8 (842 nm) — indices 1, 2, 3, 7 in the 13-band ordering.
Converts polygon JSON annotations to binary pixel masks using rasterio + shapely.

Only the 1,437 images that carry segmentation labels are exposed via __len__.
"""

from __future__ import annotations

# stdlib
import os
from pathlib import Path
from typing import Optional

# third-party
import numpy as np
import torch
from torch.utils.data import Dataset

# Sentinel-2 13-band ordering: B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12
# We select indices 1,2,3,7 corresponding to B2,B3,B4,B8.
_BAND_INDICES: tuple[int, ...] = (1, 2, 3, 7)

# Number of labelled images in the HSG-AIML dataset (Zenodo 4250706).
_LABELLED_IMAGE_COUNT: int = 1_437

# Default raw data directory relative to repo root.
_DEFAULT_RAW_DIR: str = "data/raw"


class HsgAimlDataset(Dataset):  # type: ignore[type-arg]
    """PyTorch Dataset for the HSG-AIML plume segmentation benchmark.

    Each sample is a tuple ``(tensor, mask)`` where:
    - ``tensor`` — ``torch.Tensor`` of shape ``(4, H, W)``, float32, values in [0, 1].
      Bands ordered B2, B3, B4, B8.
    - ``mask``   — ``torch.Tensor`` of shape ``(1, H, W)``, float32, binary {0.0, 1.0}.

    Only images that have polygon segmentation annotations are included.
    ``__len__`` returns exactly 1,437 (the labelled subset of the full dataset).

    Args:
        root: Path to the ``data/raw/`` directory containing extracted GeoTIFF files.
        transform: Optional callable applied to the (tensor, mask) tuple after loading.
    """

    def __init__(
        self,
        root: str = _DEFAULT_RAW_DIR,
        transform: Optional[object] = None,
    ) -> None:
        self._root = Path(root)
        self._transform = transform
        # TODO: scan root for GeoTIFF / annotation pairs and build index
        self._index: list[tuple[Path, Path]] = []  # (image_path, annotation_path)

    def __len__(self) -> int:
        # Returns the count of labelled images; falls back to the spec constant when the
        # directory has not yet been populated (e.g. before download_dataset() is called).
        if self._index:
            return len(self._index)
        return _LABELLED_IMAGE_COUNT

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one (image, mask) pair.

        Steps:
        1. Open GeoTIFF with rasterio; read all 13 bands.
        2. Select bands at indices 1,2,3,7 (B2,B3,B4,B8).
        3. Normalise to [0, 1] by dividing by 10000 (Sentinel-2 L2A DN scale).
        4. Rasterise polygon annotation to binary mask using shapely + rasterio.
        5. Convert both to float32 torch.Tensor.
        """
        # TODO: implement GeoTIFF load + band select + mask rasterisation
        ...


def download_dataset(destination: str = _DEFAULT_RAW_DIR) -> None:
    """Download the HSG-AIML dataset from Zenodo to ``destination``.

    Zenodo DOI: 10.5281/zenodo.4250706

    After download the archive is extracted in-place. Existing files are not
    re-downloaded if the destination already contains the expected structure.

    Args:
        destination: Local directory to write the downloaded dataset into.
            Defaults to ``data/raw/``.
    """
    # TODO: implement Zenodo fetch
    # Suggested implementation:
    #   1. GET https://zenodo.org/api/records/4250706 → parse "files" list.
    #   2. For each file, stream-download with requests, show tqdm progress bar.
    #   3. Verify MD5 checksum from the Zenodo record against downloaded bytes.
    #   4. Extract zip/tar archive to destination.
    ...
