# tools.ml_models.arch.study

**Source:** `packages/tools/src/tools/ml_models/arch/study.py`
**Kind:** module

## Purpose

This module builds an untrained ShuffleNet V2 x0.5 classifier and a DilateNet
segmentor whose first convolution matches the study channel count.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ShuffleNetClassifier` | class | One logit per tile |
| `DilateNet` | class | One logit plane at the input size |

## Inputs and outputs

`ShuffleNetClassifier.forward(x) -> Tensor` of shape ``(N, 1)``.

`DilateNet.forward(x) -> Tensor` of shape ``(N, 1, H, W)``.

## Behavior

1. ShuffleNet is constructed with ``weights=None``. Its first convolution is
   replaced so the input depth equals ``in_channels``. The final linear layer
   emits one logit.
2. DilateNet uses a dense stem of width 32, a separable body of width 64, four
   dilated blocks at rates 1, 2, 4, and 8, and output stride 4.
3. The segmentor head is bilinearly resized to the input spatial size.
4. During training, a ShuffleNet batch-norm layer with one value per channel
   uses its running mean and variance.

## Errors and faults

`ValueError` when ``in_channels`` is below 1. `TypeError` when the torchvision
stem is not a ``Conv2d``.

## Messages

None.

## Configuration

None.

## Constraints

ImageNet weights are not loaded. The segmentor is the width-32 separable
DilateNet, copied into this package.

## Related documents

- [`tools.ml_models.arch`](../arch.md)
- [`tools.ml_models.arch.dilated`](dilated.md)
- [`tools.ml_models.arch.classifier`](classifier.md)
