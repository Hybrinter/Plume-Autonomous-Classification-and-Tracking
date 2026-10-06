"""Deterministic pure metric cores and their declarative definitions.

Contains:
  - definitions: metric names, directions, formulas, and limitations.
  - inputs: validated binary score/label vectors and stable transforms.
  - classifier: binary counts, ranking, and operating curves.
  - calibration: Brier, reliability bins, and ECE diagnostics.
  - segmentation: per-image overlap/area/loss rows and pixel diagnostics.
  - localization: component matching and miss-inclusive success curves.
  - boundary: boundary distances and inclusive tolerance scores.
  - spatial: shared localization/boundary rows and frozen aggregation.
  - generalization: strata and group-level uncertainty (scaffold).

Only generalization remains a scaffold.
"""
