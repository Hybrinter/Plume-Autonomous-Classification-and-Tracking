"""Model workflows for training, export, and acceptance.

Contains:
  - dataset: finished per-source datasets, splits, and the build CLI inputs.
  - arch: segmentor and classifier network builders.
  - cli: ``python -m tools.ml_models`` command group.

Import from ``tools.ml_models.dataset`` or ``tools.ml_models.arch``. This package
does not re-export names.
"""
