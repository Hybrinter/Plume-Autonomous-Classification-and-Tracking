# tools.inference.arch.registry

**Source:** `packages/tools/src/tools/inference/arch/registry.py`
**Kind:** module

## Purpose

This module maps a train kind and architecture name to a network builder.
Importing the module does not import torch.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `DEFAULT_ARCH` | constant | Kind to default architecture name |
| `default_arch` | function | Default name for a kind |
| `resolve_arch` | function | Fill an empty name and reject unknown pairs |
| `build` | function | Construct an untrained logits graph |

## Inputs and outputs

`default_arch(kind) -> str`.

`resolve_arch(kind, arch) -> str`.

`build(kind, arch, in_channels) -> nn.Module`. Raises `ImportError` when torch
is missing. Raises `ValueError` on an unknown pair.

## Behavior

1. Empty `arch` selects `resnet50` for the classifier and `unet` for the
   segmentor.
2. `build` imports the matching constructor only when called.

## Errors and faults

`ValueError` on an unknown kind or architecture name. `ImportError` when torch
is not installed.

## Messages

None.

## Configuration

Known pairs: `classifier/resnet50` and `segmentor/unet`.

## Constraints

Importing this module does not import torch. Adding a family is one builder
plus one registry row.

## Related documents

- [`tools.inference.arch`](../arch.md)
- [`tools.inference.train`](../train.md)
- [`tools.inference.export`](../export.md)
