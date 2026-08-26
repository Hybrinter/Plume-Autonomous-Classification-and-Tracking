"""4-channel smoke plume classification Dataset.

Expects the layout produced by ``scripts/prepare_dataset.py``::

    <root>/classification/{train,val,test}/{positive,negative}/*.tif
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import rasterio as rio
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from smoke_detection.common.paths import classification_split
from smoke_detection.data.transforms import Normalize, RandomCrop, Randomize, ToTensor


class SmokePlumeDataset(Dataset):
    """Per-class-folder dataset for smoke plume classification.

    :param datadir: root containing ``positive`` and ``negative`` subdirs; if
        ``None``, defaults to ``classification_split('train')``.
    :param mult: multiply dataset length by this factor (for oversampling).
    :param transform: callable applied to each sample dict.
    :param balance: ``"upsample"``, ``"downsample"``, or anything else (no-op).
    """

    def __init__(
        self,
        datadir: str | Path | None = None,
        mult: int = 1,
        transform=None,
        balance: str = "upsample",
    ):
        if datadir is None:
            datadir = classification_split("train")
        self.datadir = Path(datadir)
        self.transform = transform

        imgfiles: list[str] = []
        labels: list[bool] = []
        positive_indices: list[int] = []
        negative_indices: list[int] = []

        idx = 0
        for root, _dirs, files in os.walk(self.datadir):
            for filename in files:
                if not filename.endswith(".tif"):
                    continue
                imgfiles.append(os.path.join(root, filename))
                if "positive" in root:
                    labels.append(True)
                    positive_indices.append(idx)
                    idx += 1
                elif "negative" in root:
                    labels.append(False)
                    negative_indices.append(idx)
                    idx += 1

        self.imgfiles = np.array(imgfiles)
        self.labels = np.array(labels)
        self.positive_indices = np.array(positive_indices, dtype=np.intp)
        self.negative_indices = np.array(negative_indices, dtype=np.intp)

        if balance == "downsample":
            self._balance_downsample()
        elif balance == "upsample":
            self._balance_upsample()

        if mult > 1:
            self.imgfiles = np.array([*self.imgfiles] * mult)
            self.labels = np.array([*self.labels] * mult)
            self.positive_indices = np.array([*self.positive_indices] * mult)
            self.negative_indices = np.array([*self.negative_indices] * mult)

    def __len__(self) -> int:
        return len(self.imgfiles)

    def _balance_downsample(self) -> None:
        idc = np.ravel(
            [
                self.positive_indices,
                self.negative_indices[
                    np.random.randint(0, len(self.negative_indices), len(self.positive_indices))
                ],
            ]
        ).astype(int)
        self.imgfiles = self.imgfiles[idc]
        self.labels = self.labels[idc]
        self.positive_indices = np.arange(len(self.labels))[self.labels == True]  # noqa: E712
        self.negative_indices = np.arange(len(self.labels))[self.labels == False]  # noqa: E712

    def _balance_upsample(self) -> None:
        if len(self.positive_indices) == 0 or len(self.negative_indices) == 0:
            return

        positive_indices = np.asarray(self.positive_indices, dtype=np.intp)
        extra = np.random.randint(
            0,
            len(positive_indices),
            max(0, len(self.negative_indices) - len(positive_indices)),
        ).astype(np.intp)
        extra_idc = np.asarray(positive_indices[extra], dtype=np.intp)
        self.imgfiles = np.concatenate([self.imgfiles, self.imgfiles[extra_idc]])
        self.labels = np.concatenate([self.labels, self.labels[extra_idc]])
        self.positive_indices = np.arange(len(self.labels), dtype=np.intp)[self.labels == True]  # noqa: E712
        self.negative_indices = np.arange(len(self.labels), dtype=np.intp)[self.labels == False]  # noqa: E712

    def __getitem__(self, idx: int) -> dict:
        imgfile = rio.open(self.imgfiles[idx])
        # Sentinel-2 bands: B2(490), B3(560), B4(665), B8(842)
        imgdata = np.array([imgfile.read(i) for i in [2, 3, 4, 8]], dtype=np.float32)
        imgdata = _pad_to_120(imgdata)

        sample = {
            "idx": idx,
            "img": imgdata,
            "lbl": bool(self.labels[idx]),
            "imgfile": str(self.imgfiles[idx]),
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


def _pad_to_120(imgdata: np.ndarray) -> np.ndarray:
    """Right-/bottom-pad (by repeating the last row/col) to enforce 120x120."""
    if imgdata.shape[1] != 120:
        new = np.empty((imgdata.shape[0], 120, imgdata.shape[2]), dtype=imgdata.dtype)
        new[:, : imgdata.shape[1], :] = imgdata
        new[:, imgdata.shape[1] :, :] = imgdata[:, imgdata.shape[1] - 1 :, :]
        imgdata = new
    if imgdata.shape[2] != 120:
        new = np.empty((imgdata.shape[0], 120, 120), dtype=imgdata.dtype)
        new[:, :, : imgdata.shape[2]] = imgdata
        new[:, :, imgdata.shape[2] :] = imgdata[:, :, imgdata.shape[2] - 1 :]
        imgdata = new
    return imgdata


def build_default_transform(crop_size: int = 90) -> Compose:
    """Default training-time transform pipeline."""
    return Compose([Normalize(), RandomCrop(crop=crop_size), Randomize(), ToTensor()])


def build_eval_transform() -> Compose:
    """Eval-time transform pipeline (no random augmentation)."""
    return Compose([Normalize(), ToTensor()])
