"""Processed-pack metadata, Zenodo reads, prism chips, and flight tiles.

Contains:
  - meta: dataset identity, provenance, and the pack hash.
  - pack: on-disk arrays and in-memory concatenation.
  - split: group-wise train, val, and test indices.
  - norm: DN, unit-interval, and per-band z-score recipes.
  - bands: Sentinel-2 ids and subset selection.
  - grid: legal-side coarsening and any-side area resample.
  - matrix: native band matrix for the Zenodo study.
  - zenodo: archive index, tile cache, and location splits.
  - prism: AP-3200T weights, the 76 px chip pack, and stored 193 by 258 tiles.
  - augment: dihedral transforms and feathered paste.
  - canvas: tile-sized scenes and optional windows.
"""
