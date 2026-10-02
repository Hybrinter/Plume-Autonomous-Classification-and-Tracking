"""Plain-torch training for GSD-conditioned models on finished datasets.

Contains:
  - config: frozen TrainConfig, strict TOML load, overlay, and digest.
  - loop: run directory, epoch loop, evaluation cadence, and checkpoints.
  - provenance: training geometry and cross-dataset split-leakage checks.
  - evaluate: exhaustive split scoring with dataset-weighted macro averaging.
  - losses: BCE, Dice, and focal objectives.
  - metrics: classifier and segmentor scores.

Import each module by name. This package does not re-export names.
"""
