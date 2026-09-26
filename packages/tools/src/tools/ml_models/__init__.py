"""Model workflows for training, export, analysis, and acceptance.

Contains:
  - data: processed packs, Zenodo reads, fetch, prism proxy, and flight canvas.
  - arch: segmentor and classifier network builders.
  - train: plain-torch loop, losses, metrics, cost, and sweeps.
  - export: ONNX logits, acceptance, and the flight pair blob.
  - analysis: figures, catalogs, full-frame scores, and held-out eval.
  - cli: Typer commands for train, eval, export, accept, and pair.

Import from ``tools.ml_models.data``, ``tools.ml_models.arch``,
``tools.ml_models.train``, ``tools.ml_models.export``, or
``tools.ml_models.analysis``. This package does not re-export names.
"""
