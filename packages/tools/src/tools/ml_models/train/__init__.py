"""Training configuration, provenance, and the evidence-recording loop.

Contains:
  - config: frozen TrainConfig, strict TOML load, overlay, digest, and
    validation-metric aliases.
  - loop: public train boundary; durable run evidence and checkpoints.
  - provenance: training geometry and cross-dataset split-leakage checks.
  - losses: BCE, Dice, and focal objectives.

Import each module by name. This package does not re-export names.
"""
