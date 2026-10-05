"""Model workflows for training, export, and acceptance.

Contains:
  - dataset: finished per-source datasets, splits, and the build CLI inputs.
  - arch: segmentor and classifier network builders.
  - train: training boundary (unavailable scaffold) and provenance.
  - export: two-input ONNX export, validation, acceptance, and pair gating.
  - analysis: evidence contracts, unavailable analysis boundaries, and pure
    frontier/formatting helpers.
  - studies: offline band and GSD studies over the shared dataset sources.
  - cli: ``python -m tools.ml_models`` command group.

Import from ``tools.ml_models.dataset`` or ``tools.ml_models.arch``. This package
does not re-export names.
"""
