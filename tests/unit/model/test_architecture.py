"""Unit tests for pact.model.architecture.build_model().

Satisfies: §6.2 of PACT_SW_ARCH.md — Model subsystem unit tests.
REQ-AIML-HIGH-001, REQ-AIML-HIGH-002, REQ-AIML-IMAG-001
"""

from __future__ import annotations

# third-party
import pytest
import segmentation_models_pytorch as smp
import torch

# module under test
from pact.model.architecture import build_model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_model_returns_unet() -> None:
    """build_model() must return an smp.Unet instance."""
    model = build_model(encoder_weights=None)
    assert isinstance(model, smp.Unet), (
        f"Expected smp.Unet, got {type(model)}"
    )


def test_model_output_shape() -> None:
    """Forward pass with (1,4,256,256) input must produce (1,1,256,256) output."""
    model = build_model(encoder_weights=None)
    model.eval()
    x = torch.zeros(1, 4, 256, 256, dtype=torch.float32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 256, 256), (
        f"Expected output shape (1,1,256,256), got {tuple(out.shape)}"
    )


def test_build_model_random_init() -> None:
    """build_model(encoder_weights=None) must complete without error.

    Verifies that random (untrained) initialisation works for test environments
    where ImageNet weights cannot be downloaded.
    """
    # Should not raise ImportError, RuntimeError, or any other exception.
    model = build_model(encoder_weights=None)
    assert model is not None
