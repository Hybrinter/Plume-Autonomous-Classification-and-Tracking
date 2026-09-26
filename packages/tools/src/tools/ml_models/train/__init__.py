"""Plain-torch training for classifier and segmentor runs.

Contains:
  - config: frozen hyperparameters, TOML overlay, and the channel-count rule.
  - loop: single-pack batches and mixed chip and tile steps.
  - recipe: index split recipe and the legacy pack hash.
  - samples: synthetic scenes and torch pack loaders.
  - tiles: study tiles at one band subset and side.
  - study: shared AdamW loop for the band study.
  - native_sweep: native-resolution study sweep.
  - losses: BCE, Dice, and focal objectives.
  - metrics: classifier and segmentor scores.
  - cost: parameter and FLOP counts.
  - sweep: cartesian search over train configs.

Import each module by name. This package does not re-export names.
"""
