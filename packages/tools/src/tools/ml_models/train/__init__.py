"""Training configuration, provenance, and the unavailable training boundary.

Contains:
  - config: frozen TrainConfig, strict TOML load, overlay, and digest.
  - loop: public train boundary; unavailable until evidence training lands.
  - provenance: training geometry and cross-dataset split-leakage checks.
  - losses: BCE, Dice, and focal objectives.

Import each module by name. This package does not re-export names.
"""
