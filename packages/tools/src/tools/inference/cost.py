"""Parameter and FLOP counts for an untrained or trained logits graph.

Public names are defined in ``tools.ml_models.train.cost``.

Contains:
  - count_params: number of trainable and frozen parameters.
  - count_flops: FLOPs for one forward pass at a given NCHW shape.
"""

from __future__ import annotations

from tools.ml_models.train.cost import count_flops, count_params

__all__ = ["count_flops", "count_params"]
