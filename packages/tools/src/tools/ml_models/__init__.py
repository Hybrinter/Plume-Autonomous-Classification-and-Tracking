"""Model workflows for training, export, and acceptance.

Contains:
  - dataset: finished per-source datasets, splits, and the build CLI inputs.
  - arch: segmentor and classifier network builders.
  - train: conditioned training loop, evaluation, and provenance.
  - export: two-input ONNX export, validation, acceptance, and pair gating.
  - analysis: run accounting, Pareto ranking, and report rendering.
  - studies: offline band and GSD studies over the shared dataset sources.
  - cli: ``python -m tools.ml_models`` command group.

Import from ``tools.ml_models.dataset`` or ``tools.ml_models.arch``. This package
does not re-export names.
"""
