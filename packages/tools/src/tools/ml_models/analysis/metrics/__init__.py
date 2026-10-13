"""Deterministic pure metric cores and their declarative definitions.

Contains:
  - definitions: metric names, directions, formulas, and limitations.
  - inputs: validated binary score/label vectors and stable transforms.
  - classifier: binary counts, ranking, and operating curves.
  - calibration: Brier, reliability bins, and ECE diagnostics.
  - segmentation: overlap, pixel, area, and loss records (scaffold).
  - localization: component matching and localization records (scaffold).
  - boundary: boundary distances and tolerance scores (scaffold).
  - generalization: strata and group-level uncertainty (scaffold).

Only classifier, calibration, inputs, and definitions are implemented.
"""
