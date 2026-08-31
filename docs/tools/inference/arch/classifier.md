# tools.inference.arch.classifier

**Source:** `packages/tools/src/tools/inference/arch/classifier.py`
**Kind:** module

## Purpose

This module builds binary plume classifiers on torchvision backbones. The stem
accepts four input bands. The head emits one logit. A trailing `_pt` suffix
loads ImageNet weights and remaps the stem. The empty-arch default classifier
is compact `pactnet`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `PRETRAINED_SUFFIX` | constant | Suffix string `_pt` |
| `BackboneName` | enum | Registered torchvision backbone names |
| `CLASSIFIER_BACKBONES` | constant | String values of `BackboneName` |
| `RESNET_BACKBONES` | constant | ResNet members of `BackboneName` |
| `BackboneSpec` | class | Backbone name and pretrained flag |
| `parse_backbone` | function | Split a registry name into a `BackboneSpec` |
| `construct_backbone` | function | Stock torchvision network for a spec |
| `build_backbone_spec` | function | Retarget a parsed spec to PACT bands |
| `build_backbone` | function | Construct a retargeted torchvision classifier |
| `build_classifier` | function | Construct the default pactnet classifier |

## Inputs and outputs

`parse_backbone(name) -> BackboneSpec`. Raises `ValueError` on an unknown
backbone.

`construct_backbone(spec) -> nn.Module`. Returns the unmodified torchvision
network, still RGB and 1000-way.

`build_backbone_spec(spec, in_channels=4) -> nn.Module`. Forward maps
`(N, C, H, W)` to `(N, 1)` logits.

`build_backbone(name, in_channels=4) -> nn.Module`. Parses `name` then calls
`build_backbone_spec`.

`build_classifier(in_channels=4) -> nn.Module`. Returns an untrained PactNet.

## Behavior

1. Registered backbones: `resnet18`, `resnet34`, `resnet50`, `mobilenetv3_small`,
   `mobilenetv3_large`, `efficientnet_b0`, `shufflenetv2_x0_5`.
2. `parse_backbone` strips `_pt` and constructs `BackboneName(raw)`.
3. `construct_backbone` matches on `BackboneName` and calls the torchvision
   constructor.
4. `build_backbone_spec` retargets the first convolution to `in_channels` and
   replaces the final linear layer with one output.
5. `build_classifier` builds default-width compact `pactnet` with random
   weights.

## Errors and faults

`ValueError` on an unknown backbone name.

## Messages

None.

## Configuration

`in_channels` defaults to 4. Pretrained stem remapping uses
[`tools.inference.arch.stem`](stem.md).

## Constraints

This module imports torch and torchvision at import time. The graph does not
apply sigmoid. Registry names use the classifier grammar in
[`tools.inference.arch.registry`](registry.md). Family builders stay family
specific: `build_backbone` constructs torchvision graphs.

## Related documents

- [`tools.inference.arch`](../arch.md)
- [`tools.inference.arch.compact`](compact.md)
- [`tools.inference.arch.stem`](stem.md)
- [`tools.inference.arch.registry`](registry.md)
- [`tools.inference.train`](../train.md)
