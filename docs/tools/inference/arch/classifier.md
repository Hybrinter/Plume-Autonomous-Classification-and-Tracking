# tools.inference.arch.classifier

**Source:** `packages/tools/src/tools/inference/arch/classifier.py`
**Kind:** module

## Purpose

This module builds a torchvision ResNet-50 binary classifier. The stem accepts
four input bands. The head emits one logit.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `build_classifier` | function | Construct a 4-channel, 1-logit ResNet-50 |

## Inputs and outputs

`build_classifier(in_channels=4) -> nn.Module`.

Forward maps `(N, C, H, W)` to `(N, 1)` logits.

## Behavior

1. Construct `resnet50(weights=None)`.
2. Replace `conv1` with an `in_channels` stem. Keep the 7x7, stride-2, pad-3
   geometry.
3. Replace `fc` with a linear layer of one output.

## Errors and faults

None beyond torch runtime errors.

## Messages

None.

## Configuration

`in_channels` defaults to 4.

## Constraints

This module imports torch and torchvision at import time. ImageNet weights are
not loaded. The graph does not apply sigmoid.

## Related documents

- [`tools.inference.arch`](../arch.md)
- [`tools.inference.train`](../train.md)
