"""Processed-pack metadata, splits, and normalization.

Contains:
  - meta: dataset identity, provenance, and the pack hash.
  - pack: on-disk arrays and in-memory concatenation.
  - split: group-wise train, val, and test indices.
  - norm: DN, unit-interval, and per-band z-score recipes.
"""
