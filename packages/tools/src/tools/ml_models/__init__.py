"""Model workflows for training, export, and acceptance.

Contains:
  - data: processed packs, Zenodo reads, prism chips, and flight tiles.
  - arch: segmentor and classifier network builders.
  - train: plain-torch loop, losses, metrics, cost, and sweeps.

Import from ``tools.ml_models.data``, ``tools.ml_models.arch``, or
``tools.ml_models.train``. This package does not re-export names.
"""
