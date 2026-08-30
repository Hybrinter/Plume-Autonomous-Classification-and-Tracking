# tools.inference.arch.classifier

**Source:** `packages/tools/src/tools/inference/arch/classifier.py`
**Kind:** module

## Purpose

This module builds binary plume classifiers on torchvision backbones. The stem
accepts four input bands. The head emits one logit. A trailing `_pt` suffix
loads ImageNet weights and remaps the stem.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `PRETRAINED_SUFFIX` | constant | Suffix string `_pt` |
| `CLASSIFIER_BACKBONES` | constant | Registered backbone names |
| `BackboneSpec` | class | Backbone name and pretrained flag |
| `parse_backbone` | function | Split a registry name into a `BackboneSpec` |
| `build_backbone` | function | Construct a retargeted classifier |
| `build_classifier` | function | Construct the default ResNet-50 classifier |

## Inputs and outputs

`parse_backbone(name) -> BackboneSpec`. Raises `ValueError` on an unknown
backbone.

`build_backbone(name, in_channels=4) -> nn.Module`. Forward maps
`(N, C, H, W)` to `(N, 1)` logits.

`build_classifier(in_channels=4) -> nn.Module`. Returns untrained ResNet-50.

## Behavior

1. Registered backbones: `resnet18`, `resnet34`, `resnet50`, `mobilenetv3_small`,
   `mobilenetv3_large`, `efficientnet_b0`, `shufflenetv2_x0_5`.
2. `parse_backbone` strips `_pt` and sets the pretrained flag.
3. `build_backbone` constructs the torchvision network, retargets the first
   convolution to `in_channels`, and replaces the final linear layer with one
   output.
4. `build_classifier` calls `build_backbone("resnet50")` with random weights.

## Errors and faults

`ValueError` on an unknown backbone name.

## Messages

None.

## Configuration

`in_channels` defaults to 4. Pretrained stem remapping uses
[`tools.inference.arch.stem`](stem.md).

## Constraints

This module imports torch and torchvision at import time. The graph does not apply
sigmoid. Registry names use the classifier grammar in
[`tools.inference.arch.registry`](registry.md).

## Related documents

- [`tools.inference.arch`](../arch.md)
- [`tools.inference.arch.stem`](stem.md)
- [`tools.inference.arch.registry`](registry.md)
- [`tools.inference.train`](../train.md)
