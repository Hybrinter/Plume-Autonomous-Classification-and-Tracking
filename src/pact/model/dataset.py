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
import json
import pathlib
import urllib.request
import zipfile
from typing import Optional

# third-party
import numpy as np
import rasterio
import rasterio.features
import torch
from torch.utils.data import Dataset

# Sentinel-2 13-band ordering: B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12
# We select indices 1,2,3,7 corresponding to B2,B3,B4,B8.
_BAND_INDICES: tuple[int, ...] = (1, 2, 3, 7)

# Rasterio uses 1-based band indexing; these correspond to _BAND_INDICES.
_RASTERIO_BANDS: list[int] = [2, 3, 4, 8]

# Number of labelled images in the HSG-AIML dataset (Zenodo 4250706).
_LABELLED_IMAGE_COUNT: int = 1_437

# Sentinel-2 L2A digital number scale factor.
_S2_DN_SCALE: float = 10_000.0

# Default raw data directory relative to repo root.
_DEFAULT_RAW_DIR: str = "data/raw"

# Zenodo download URL for the HSG-AIML dataset archive.
_ZENODO_URL: str = (
    "https://zenodo.org/record/4250706/files/hsg-aiml.zip"
)


class HsgAimlDataset(Dataset):  # type: ignore[type-arg]
    """PyTorch Dataset for the HSG-AIML plume segmentation benchmark.

    Each sample is a tuple ``(tensor, mask)`` where:
    - ``tensor`` — ``torch.Tensor`` of shape ``(4, H, W)``, float32,
      values in [0, 1]. Bands ordered B2, B3, B4, B8.
    - ``mask``   — ``torch.Tensor`` of shape ``(1, H, W)``, float32,
      binary {0.0, 1.0}.

    Only images that have polygon segmentation annotations are included.

    Args:
        root: Path to the ``data/raw/`` directory containing extracted
            GeoTIFF files.
        transform: Optional albumentations-compatible callable applied
            to the (image, mask) pair after loading.
    """

    def __init__(
        self,
        root: str = _DEFAULT_RAW_DIR,
        transform: Optional[object] = None,
    ) -> None:
        self._root = pathlib.Path(root)
        self._transform = transform

        # Scan for matching image/label pairs.
        images_dir = self._root / "images"
        labels_dir = self._root / "labels"

        self._image_files: list[pathlib.Path] = sorted(
            images_dir.glob("*.tif")
        ) if images_dir.is_dir() else []

        self._label_files: list[pathlib.Path] = sorted(
            labels_dir.glob("*.json")
        ) if labels_dir.is_dir() else []

        # Build paired index: only include images with a matching label.
        self._index: list[tuple[pathlib.Path, pathlib.Path]] = []
        label_stems = {p.stem: p for p in self._label_files}
        for img_path in self._image_files:
            label_path = label_stems.get(img_path.stem)
            if label_path is not None:
                self._index.append((img_path, label_path))

    def __len__(self) -> int:
        if self._index:
            return len(self._index)
        return _LABELLED_IMAGE_COUNT

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one (image, mask) pair.

        Steps:
        1. Open GeoTIFF with rasterio; read bands B2, B3, B4, B8.
        2. Normalise to [0, 1] by dividing by 10000 (Sentinel-2 L2A DN).
        3. Rasterise polygon annotation to binary mask.
        4. Convert both to float32 torch.Tensor.
        """
        img_path, label_path = self._index[idx]

        # Load selected spectral bands from GeoTIFF.
        with rasterio.open(img_path) as src:
            # shape (4, H, W), dtype uint16
            data = src.read(_RASTERIO_BANDS)
            transform = src.transform
            height, width = src.height, src.width

        # Normalise to [0, 1].
        image = data.astype(np.float32) / _S2_DN_SCALE  # (4, H, W)

        # Load GeoJSON label and rasterise to binary mask.
        with open(label_path) as f:
            label_data = json.load(f)

        shapes = [
            (feature["geometry"], 1)
            for feature in label_data.get("features", [])
        ]
        if shapes:
            mask = rasterio.features.rasterize(
                shapes,
                out_shape=(height, width),
                transform=transform,
                dtype=np.uint8,
            )
        else:
            mask = np.zeros((height, width), dtype=np.uint8)

        mask = mask[np.newaxis, :, :]  # (1, H, W)

        # Convert to tensors.
        image_tensor = torch.from_numpy(image)  # (4, H, W) float32
        mask_tensor = torch.from_numpy(
            mask.astype(np.float32)
        )  # (1, H, W) float32

        # Apply optional augmentation transform (albumentations API).
        if self._transform is not None:
            img_hwc = image_tensor.permute(1, 2, 0).numpy()
            mask_hw = mask_tensor[0].numpy()
            transformed = self._transform(  # type: ignore[operator]
                image=img_hwc, mask=mask_hw
            )
            image_tensor = torch.from_numpy(
                transformed["image"]
            ).permute(2, 0, 1)
            mask_tensor = torch.from_numpy(
                transformed["mask"]
            ).unsqueeze(0)

        return image_tensor, mask_tensor


def download_dataset(destination: str = _DEFAULT_RAW_DIR) -> None:
    """Download the HSG-AIML dataset from Zenodo to ``destination``.

    Zenodo DOI: 10.5281/zenodo.4250706

    After download the archive is extracted in-place. Existing files are
    not re-downloaded if the destination already contains the expected
    structure.

    Args:
        destination: Local directory to write the downloaded dataset
            into. Defaults to ``data/raw/``.
    """
    root_path = pathlib.Path(destination)
    root_path.mkdir(parents=True, exist_ok=True)
    zip_path = root_path / "hsg-aiml.zip"

    # Skip download if images directory already exists.
    if (root_path / "images").is_dir():
        return

    # Stream download with progress reporting.
    import tqdm  # noqa: E402 — optional dep, import at use site

    with urllib.request.urlopen(_ZENODO_URL) as response:
        total = int(response.headers.get("Content-Length", 0))
        with (
            open(zip_path, "wb") as f,
            tqdm.tqdm(
                total=total, unit="B", unit_scale=True
            ) as pbar,
        ):
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))

    # Extract and clean up.
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root_path)
    zip_path.unlink()
