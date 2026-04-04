"""
pact.model.architecture — U-Net/ResNet-34 model factory.

Satisfies: REQ-AIML-HIGH-001, REQ-AIML-HIGH-002, REQ-AIML-IMAG-001

The segmentation model takes a (batch, 4, H, W) float32 tensor containing Sentinel-2
bands B2 (490 nm), B3 (560 nm), B4 (665 nm), and B8 (842 nm), and returns a
(batch, 1, H, W) float32 logit map. Sigmoid is applied externally at inference time only.

Note on 4-channel input: ResNet-34 was pretrained on 3-channel ImageNet data.
`segmentation_models_pytorch` handles arbitrary `in_channels` by adapting (replacing) the
first convolutional layer of the encoder. When `encoder_weights="imagenet"` is used the
remaining encoder weights are still loaded from the pretrained checkpoint; only the first
conv layer is re-initialized. This is the standard transfer-learning strategy for
multispectral inputs and is documented in the smp library.
"""

from __future__ import annotations

# stdlib
from typing import Optional

# third-party
import segmentation_models_pytorch as smp


def build_model(encoder_weights: Optional[str] = "imagenet") -> smp.Unet:
    """Build a U-Net with a ResNet-34 encoder for 4-band VNIR plume segmentation.

    Args:
        encoder_weights: ``"imagenet"`` for ImageNet-pretrained encoder weights, or
            ``None`` for random initialisation (e.g. during unit tests).

    Returns:
        An ``smp.Unet`` instance configured for:
        - Input:  (batch, 4, H, W) float32  — B2, B3, B4, B8 bands
        - Output: (batch, 1, H, W) float32  — raw logits (sigmoid applied externally)

    Satisfies: REQ-AIML-HIGH-001, REQ-AIML-HIGH-002
    """
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=encoder_weights,
        in_channels=4,       # B2, B3, B4, B8 — first conv layer adapted by smp
        classes=1,           # binary plume mask
        activation=None,     # raw logits; sigmoid applied externally at inference time
    )
