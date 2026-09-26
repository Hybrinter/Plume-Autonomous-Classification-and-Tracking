"""Figures, run catalogs, held-out scores, and full-frame evaluation.

Contains:
  - plots: headless matplotlib writers.
  - report: figures and a markdown summary for one run.
  - runs: discovery and text tables for the local catalog.
  - pareto: size against quality, including flight hit rate and val Dice.
  - eval: checkpoint scoring on a named split.
  - results: native and ground-sample markdown tables.
  - native: coarse logits scored on the 120 px mask.
  - full_frame: blob hit, empty-frame false positives, and eval scenes.

Import each module by name. This package does not re-export names.
"""
