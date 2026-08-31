# tools.inference.arch.stem

**Source:** `packages/tools/src/tools/inference/arch/stem.py`
**Kind:** module

## Purpose

This module retargets ImageNet-pretrained convolution stems to the PACT band
count. Torchvision backbones expect three RGB planes. PACT feeds four bands in
BLUE, GREEN, RED, NIR order.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `BAND_TO_RGB_INDEX` | constant | PACT band position to pretrained RGB kernel column |
| `remap_stem_weight` | function | Build an `in_channels` kernel from an RGB kernel |
| `adapt_conv_in_channels` | function | Replace a `Conv2d` input band count |
| `retarget_first_conv` | function | Rewrite a backbone stem to accept `in_channels` planes |
| `retarget_final_linear` | function | Rewrite a backbone head to emit one logit |

## Inputs and outputs

`remap_stem_weight(weight, in_channels) -> Tensor`. Returns
`(out, in_channels, kh, kw)`. Raises `ValueError` on a non-RGB kernel or
invalid `in_channels`.

`adapt_conv_in_channels(conv, in_channels, pretrained) -> Conv2d`.

`retarget_first_conv(model, in_channels, pretrained) -> None`. Raises
`ValueError` when the model has no convolution.

`retarget_final_linear(model, out_features) -> None`. Raises `ValueError` when
the model has no linear layer.

## Behavior

1. `BAND_TO_RGB_INDEX` maps BLUE, GREEN, and RED to the pretrained blue, green,
   and red kernel columns.
2. Extra bands beyond three take the mean RGB column.
3. The remapped kernel scales by `3 / in_channels` to preserve activation
   magnitude for downstream batch-norm statistics.
4. `retarget_first_conv` locates the first `Conv2d` by module traversal.
5. `retarget_final_linear` replaces the last `Linear` layer.

## Errors and faults

`ValueError` on an invalid kernel shape, `in_channels` below one, or a model
with no stem convolution or linear head.

## Messages

None.

## Configuration

PACT band order is BLUE, GREEN, RED, NIR. Flight default `in_channels` is 4.

## Constraints

This module imports torch. Classifier and encoder-segmentor builders call these
helpers when a `_pt` or `pt` suffix requests ImageNet weights.

## Related documents

- [`tools.inference.arch`](../arch.md)
- [`tools.inference.arch.classifier`](classifier.md)
- [`tools.inference.arch.encoder_unet`](encoder_unet.md)
