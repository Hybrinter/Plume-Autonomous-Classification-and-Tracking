# tools.ml_models.arch.stem

**Source:** `packages/tools/src/tools/ml_models/arch/stem.py`
**Kind:** module

## Purpose

This module retargets ImageNet-pretrained convolution stems to a chosen band
count. Torchvision backbones expect three RGB planes. The flight stem is three
channels in BLUE, GREEN, RED order. `remap_stem_weight` copies an RGB kernel
onto that prefix.

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

Flight band order is BLUE, GREEN, RED. Flight default `in_channels` is 3.
Bands past the third take the mean RGB column.

## Constraints

This module imports torch. Classifier and encoder-segmentor builders call these
helpers when a `_pt` or `pt` suffix requests ImageNet weights.

## Related documents

- [`tools.ml_models.arch`](../arch.md)
- [`tools.ml_models.arch.classifier`](classifier.md)
- [`tools.ml_models.arch.encoder_unet`](encoder_unet.md)
