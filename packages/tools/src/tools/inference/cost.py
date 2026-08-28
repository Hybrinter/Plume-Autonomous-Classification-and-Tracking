"""Parameter and FLOP counts for an untrained or trained logits graph.

Counts use torch only. FLOPs come from ``FlopCounterMode`` over one dummy
batch at the configured spatial size.

Contains:
  - count_params: number of trainable and frozen parameters.
  - count_flops: FLOPs for one forward pass at a given NCHW shape.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode


def count_params(model: nn.Module) -> int:
    """Return the number of parameters in ``model``.

    Args:
        model: A torch module.

    Returns:
        int: Sum of ``numel()`` over all parameters.
    """
    return int(sum(item.numel() for item in model.parameters()))


def count_flops(model: nn.Module, input_shape: tuple[int, ...]) -> int:
    """Return FLOPs for one forward pass at ``input_shape``.

    Args:
        model: A torch module that maps NCHW tensors to logits.
        input_shape: Dummy input shape, for example ``(1, 4, 256, 256)``.

    Returns:
        int: Total FLOPs recorded by ``FlopCounterMode``.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.zeros(input_shape, dtype=torch.float32, device=device)
    with FlopCounterMode(display=False) as flop:
        with torch.no_grad():
            model(dummy)
    if was_training:
        model.train()
    return int(flop.get_total_flops())
