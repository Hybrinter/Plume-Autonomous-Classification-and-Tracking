"""4-channel smoke plume segmentation Dataset.

Expects the layout produced by ``scripts/prepare_dataset.py``::

    <root>/segmentation/{train,val,test}/images/{positive,negative}/*.tif
    <root>/segmentation/{train,val,test}/labels/*.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.features import rasterize
from shapely.geometry import Polygon
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from smoke_detection.common.paths import segmentation_split
from smoke_detection.data.classification_dataset import _pad_to_120
from smoke_detection.data.transforms import Normalize, RandomCrop, Randomize, ToTensor


def label_image_url_to_tif_key(image_url: str) -> str:
    """Map a Label Studio image URL to the on-disk GeoTIFF basename."""
    key = "-".join(image_url.split("-")[1:]).replace(".png", ".tif")
    return key.replace(":", "_")


class SmokePlumeSegmentationDataset(Dataset):
    """Paired image + rasterized polygon-mask dataset."""

    def __init__(
        self,
        datadir: str | Path | None = None,
        seglabeldir: str | Path | None = None,
        mult: int = 1,
        transform=None,
    ):
        if datadir is None or seglabeldir is None:
            di, dl = segmentation_split("train")
            datadir = datadir if datadir is not None else di
            seglabeldir = seglabeldir if seglabeldir is not None else dl

        self.datadir = Path(datadir)
        self.transform = transform

        imgfiles: list[str] = []
        labels: list[bool] = []
        seglabels_per_img: list[list[np.ndarray]] = []
        positive_indices: list[int] = []
        negative_indices: list[int] = []

        raw_seglabels = []
        segfile_lookup: dict[str, int] = {}
        for i, seglabelfile in enumerate(os.listdir(seglabeldir)):
            with open(os.path.join(seglabeldir, seglabelfile)) as f:
                segdata = json.load(f)
            raw_seglabels.append(segdata)
            segfile_lookup[label_image_url_to_tif_key(segdata["data"]["image"])] = i

        idx = 0
        for root, _dirs, files in os.walk(self.datadir):
            for filename in files:
                if not filename.endswith(".tif"):
                    continue
                if filename not in segfile_lookup:
                    continue
                polygons: list[np.ndarray] = []
                for completions in raw_seglabels[segfile_lookup[filename]]["completions"]:
                    for result in completions["result"]:
                        pts = result["value"]["points"] + [result["value"]["points"][0]]
                        polygons.append(np.array(pts) * 1.2)
                if "positive" in root and polygons:
                    labels.append(True)
                    positive_indices.append(idx)
                    imgfiles.append(os.path.join(root, filename))
                    seglabels_per_img.append(polygons)
                    idx += 1

        # Pair each positive with an equal number of negatives (mirrors original code).
        n_pos = len(positive_indices)
        for root, _dirs, files in os.walk(self.datadir):
            for filename in files:
                if idx >= 2 * n_pos:
                    break
                if not filename.endswith(".tif"):
                    continue
                if "negative" in root:
                    labels.append(False)
                    negative_indices.append(idx)
                    imgfiles.append(os.path.join(root, filename))
                    seglabels_per_img.append([])
                    idx += 1

        self.imgfiles = np.array(imgfiles)
        self.labels = np.array(labels)
        self.positive_indices = np.array(positive_indices)
        self.negative_indices = np.array(negative_indices)
        self.seglabels = seglabels_per_img

        if mult > 1:
            self.imgfiles = np.array([*self.imgfiles] * mult)
            self.labels = np.array([*self.labels] * mult)
            self.positive_indices = np.array([*self.positive_indices] * mult)
            self.negative_indices = np.array([*self.negative_indices] * mult)
            self.seglabels = self.seglabels * mult

    def __len__(self) -> int:
        return len(self.imgfiles)

    def __getitem__(self, idx: int) -> dict:
        imgfile = rio.open(self.imgfiles[idx])
        imgdata = np.array([imgfile.read(i) for i in [2, 3, 4, 8]], dtype=np.float32)
        imgdata = _pad_to_120(imgdata)

        fptdata = np.zeros(imgdata.shape[1:], dtype=np.uint8)
        polygons = self.seglabels[idx]
        shapes: list[Polygon] = []
        if polygons:
            for pol in polygons:
                try:
                    shapes.append(Polygon(pol))
                except ValueError:
                    continue
            fptdata = rasterize(((g, 1) for g in shapes), out_shape=fptdata.shape, all_touched=True)

        sample = {
            "idx": idx,
            "img": imgdata,
            "fpt": fptdata.astype(np.float32),
            "imgfile": str(self.imgfiles[idx]),
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


def build_default_transform(crop_size: int = 90) -> Compose:
    """Default training-time transform pipeline for segmentation."""
    return Compose([Normalize(), Randomize(), RandomCrop(crop=crop_size), ToTensor()])


def build_eval_transform() -> Compose:
    """Eval-time transform pipeline (no random augmentation)."""
    return Compose([Normalize(), ToTensor()])
