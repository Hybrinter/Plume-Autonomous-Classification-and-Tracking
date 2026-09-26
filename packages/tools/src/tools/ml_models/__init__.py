"""Model workflows for training, export, and acceptance.

Contains:
  - data: processed packs, Zenodo reads, prism proxy, and flight canvas.
  - arch: segmentor and classifier network builders.
  - train: plain-torch loop, losses, metrics, cost, and sweeps.
  - export: ONNX logits, acceptance, and the flight pair blob.
  - cli: Typer commands for train, eval, export, accept, and pair.

Import from ``tools.ml_models.data``, ``tools.ml_models.arch``,
``tools.ml_models.train``, or ``tools.ml_models.export``. This package does
not re-export names.
"""
