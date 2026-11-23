"""GSD conditioning for the flight model families.

GSD varies across finished datasets and across bins inside one dataset. A
pixel-aligned model that ignores it must entangle appearance with scale. The
conditioned families instead receive the log-ratio encoding produced by
``to_model_gsd`` as a second ``(N, 2)`` input, and FiLM blocks translate it
into per-channel scale and shift applied to the feature maps.

The modulation is identity at initialisation, so conditioning can only open
where gradients support it. ``CONDITIONING_ID`` marks which call signature a
checkpoint carries: the conditioned families record ``film-log-gsd-v1`` and
every other registry graph is wrapped in :class:`IgnoreGsd`, which records
``ignored``.

Contains:
  - CONDITIONING_ID: checkpoint conditioning marker for conditioned models.
  - IGNORED_CONDITIONING_ID: marker for wrapped single-input models.
  - GsdModel: protocol for an ``(image, gsd) -> logits`` graph.
  - GsdFilm: FiLM block mapping ``(N, 2)`` GSD to per-channel affine terms.
  - IgnoreGsd: adapter that discards GSD for single-input graphs.
  - resolve_gsd: the reference-encoding fallback for a missing GSD input.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn

CONDITIONING_ID = "film-log-gsd-v1"

IGNORED_CONDITIONING_ID = "ignored"

_HIDDEN_WIDTH = 16


class GsdModel(Protocol):
    """Graph callable as ``model(image, gsd)``.

    Attributes:
        None.
    """

    def __call__(self, image: torch.Tensor, gsd: torch.Tensor) -> torch.Tensor:
        """Return logits for a band stack and its encoded GSD."""
        ...


class GsdFilm(nn.Module):
    """Feature-wise linear modulation driven by encoded GSD.

    A two-layer perceptron maps the ``(N, 2)`` encoding to ``2 * channels``
    parameters, split into a residual scale ``delta_gamma`` and an additive
    ``beta``. The final linear layer is zero-initialised, so a fresh block is
    an exact identity on its features and training alone opens the terms.
    """

    def __init__(self, channels: int) -> None:
        """Build the modulation head.

        Args:
            channels: Feature-map channel count the block modulates.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, _HIDDEN_WIDTH),
            nn.ReLU(),
            nn.Linear(_HIDDEN_WIDTH, 2 * channels),
        )
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, features: torch.Tensor, gsd: torch.Tensor) -> torch.Tensor:
        """Apply GSD-conditioned scale and shift to a feature map.

        Args:
            features: Input of shape ``(N, C, H, W)``.
            gsd: Encoded GSD of shape ``(N, 2)``.

        Returns:
            torch.Tensor: Modulated features, same shape as ``features``.
        """
        params: torch.Tensor = self.net(gsd)
        delta_gamma, beta = params.chunk(2, dim=1)
        modulated: torch.Tensor = (
            features * (1.0 + delta_gamma[..., None, None]) + beta[..., None, None]
        )
        return modulated


class IgnoreGsd(nn.Module):
    """Adapt a single-input registry graph to the :class:`GsdModel` signature.

    Architectures outside the conditioned flight families keep their
    one-argument forward; this wrapper discards the GSD tensor so the training
    loop calls every registered model the same way. The GSD argument is
    optional, so legacy single-input callers such as the inference trainer and
    the ONNX exporter keep working unchanged.
    """

    def __init__(self, inner: nn.Module) -> None:
        """Wrap a single-input graph.

        Args:
            inner: Registry model called as ``inner(image)``.
        """
        super().__init__()
        self.inner = inner

    def forward(self, image: torch.Tensor, gsd: torch.Tensor | None = None) -> torch.Tensor:
        """Forward the image and discard the encoded GSD.

        Args:
            image: Input band stack.
            gsd: Encoded GSD. Ignored; may be omitted by legacy callers.

        Returns:
            torch.Tensor: ``inner(image)``.
        """
        result: torch.Tensor = self.inner(image)
        return result


def resolve_gsd(x: torch.Tensor, gsd: torch.Tensor | None) -> torch.Tensor:
    """Return ``gsd`` or the reference encoding for a single-input caller.

    Args:
        x: Input band stack of shape ``(N, C, H, W)``.
        gsd: Encoded GSD of shape ``(N, 2)``, or None.

    Returns:
        torch.Tensor: ``gsd`` when provided, else zeros on ``x``'s device and
        dtype. Zeros are the encoding of the reference GSD, so legacy
        single-input calls behave as though every tile sat at the reference.
    """
    if gsd is not None:
        return gsd
    return torch.zeros(x.shape[0], 2, dtype=x.dtype, device=x.device)
