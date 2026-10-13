"""Parameter counts and unavailable FLOP resource evidence.

Parameter counting is a pure helper over a torch module and remains usable.
The old FLOP path executed one dummy image input and cannot describe a
conditioned two-input model; it is unavailable until conditioned resource
measurement is implemented.

Contains:
  - count_params: number of parameters in a module.
  - count_flops: unavailable FLOP boundary.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from flight.libs.types import Err, Result
from torch import nn


def count_params(model: nn.Module) -> int:
    """Return the number of parameters in ``model``.

    Args:
        model: A torch module.

    Returns:
        int: Sum of ``numel()`` over all parameters.
    """
    return int(sum(item.numel() for item in model.parameters()))


def count_flops(model: nn.Module, input_shape: tuple[int, ...]) -> Result[int, str]:
    """Refuse to measure FLOPs while conditioned resource measurement is unimplemented.

    Args:
        model: A torch module.
        input_shape: Input shape the removed executor would have used.

    Returns:
        Result[int, str]: Always Err.
    """
    del model, input_shape
    return Err(
        "FLOP measurement is unavailable until the conditioned resource "
        "evidence phase is implemented"
    )
