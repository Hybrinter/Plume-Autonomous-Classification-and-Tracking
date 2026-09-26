"""Plain-torch training for classifier and segmentor runs.

Contains:
  - config: frozen hyperparameters, TOML overlay, and the channel-count rule.
  - loop: chip batches and flight-frame canvas steps.
  - losses: BCE, Dice, and focal objectives.
  - metrics: classifier and segmentor scores.
  - cost: parameter and FLOP counts.
  - sweep: cartesian search over train configs.

Import each module by name. This package does not re-export names.
"""
