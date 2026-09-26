"""Plain-torch train loop for the classifier and the segmentor.

Public names are defined in ``tools.ml_models.train``. ``train`` lives in
``tools.ml_models.train.loop``. Config helpers live in
``tools.ml_models.train.config``.

Contains:
  - TrainConfig: frozen hyperparameters.
  - load_train_config: dataclass defaults overlaid with an optional TOML file.
  - overlay_train_config: CLI field overlays.
  - apply_train_mapping: overlay from a string-key mapping.
  - config_digest: 8-hex identity of experiment fields.
  - resolve_train_channels: pack channel count used for the model build.
  - write_train_config_toml: write a TrainConfig table.
  - train: run the loop and write a run directory.
  - is_cuda_oom: detect a CUDA allocator failure.
  - next_batch_after_oom: halve a batch size, or raise at size 1.
  - fit_batch_size: lower the batch until one training step fits.
"""

from __future__ import annotations

from tools.ml_models.train.config import (
    TrainConfig,
    apply_train_mapping,
    config_digest,
    load_train_config,
    overlay_train_config,
    resolve_train_channels,
    write_train_config_toml,
)
from tools.ml_models.train.loop import fit_batch_size, is_cuda_oom, next_batch_after_oom, train

__all__ = [
    "TrainConfig",
    "apply_train_mapping",
    "config_digest",
    "fit_batch_size",
    "is_cuda_oom",
    "load_train_config",
    "next_batch_after_oom",
    "overlay_train_config",
    "resolve_train_channels",
    "train",
    "write_train_config_toml",
]
