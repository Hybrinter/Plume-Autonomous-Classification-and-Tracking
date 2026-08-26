"""Segmentation experiment config."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from smoke_detection.configs.base import BaseConfig


class SegmentationModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    architecture: Literal["unet"] = "unet"
    in_channels: int = 4
    n_classes: int = 1
    bilinear: bool = True


class SegmentationDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = 16
    num_workers: int = 4
    crop_size: int = 90


class SegmentationConfig(BaseConfig):
    task: Literal["segmentation"] = "segmentation"
    model: SegmentationModelConfig
    data: SegmentationDataConfig
