# tools.inference.arch.encoder_unet

**Source:** `packages/tools/src/tools/inference/arch/encoder_unet.py`
**Kind:** module

## Purpose

This module defines a segmentor with a torchvision ResNet encoder and a U-Net
decoder. Encoder capacity and decoder width are independent knobs. Output is a
logit mask at the input spatial size.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `RESNET_ENCODERS` | constant | Registered encoder names |
| `encoder_tap_channels` | function | Skip channel counts for one encoder |
| `decoder_widths` | function | Decoder channel counts for a width knob |
| `ResNetUNet` | class | Encoder-decoder segmentor |
| `build_encoder_segmentor` | function | Construct a `ResNetUNet` from a registry name |

## Inputs and outputs

`encoder_tap_channels(encoder) -> tuple[int, int, int, int, int]`. Raises
`ValueError` on an unknown encoder.

`decoder_widths(decoder_width) -> tuple[int, ...]`. Five stages halving from
`decoder_width * 8` down to `decoder_width`.

`ResNetUNet.forward(x)` maps `(N, C, H, W)` to `(N, 1, H, W)` logits.

`build_encoder_segmentor(encoder, in_channels=4, out_channels=1, pretrained=False,
decoder_width=16, separable=False) -> ResNetUNet`.

## Behavior

1. Encoders: `resnet18`, `resnet34`, `resnet50`.
2. The encoder keeps torchvision block names. Only the stem convolution is
   retargeted to the PACT band count.
3. Skip taps follow the ResNet stride schedule. The decoder upsamples with
   bilinear interpolation and concatenates skips.
4. `separable` selects depthwise-separable decoder convolutions.
5. The forward pass does not apply sigmoid.

## Errors and faults

`ValueError` on an unknown encoder or a `decoder_width` below one.

## Messages

None.

## Configuration

`in_channels` defaults to 4. `out_channels` defaults to 1. `decoder_width`
defaults to 16. `pretrained` loads ImageNet encoder weights and remaps the stem.

## Constraints

This module imports torch and torchvision. Registry names use the `runet18`,
`runet34`, and `runet50` families. See
[`tools.inference.arch.registry`](registry.md).

## Related documents

- [`tools.inference.arch`](../arch.md)
- [`tools.inference.arch.stem`](stem.md)
- [`tools.inference.arch.unet`](unet.md)
- [`tools.inference.arch.registry`](registry.md)
