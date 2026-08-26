"""Classification experiment config."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from smoke_detection.configs.base import BaseConfig


class ClassificationModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backbone: Literal["resnet50"] = "resnet50"
    pretrained: bool = True
    in_channels: int = 4


class ClassificationDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = 32
    num_workers: int = 4
    crop_size: int = 90
    balance: Literal["upsample", "downsample", "none"] = "upsample"


class ClassificationConfig(BaseConfig):
    task: Literal["classification"] = "classification"
    model: ClassificationModelConfig
    data: ClassificationDataConfig
