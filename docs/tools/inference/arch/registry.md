# tools.inference.arch.registry

**Source:** `packages/tools/src/tools/inference/arch/registry.py`
**Kind:** module

## Purpose

This module maps a train kind and architecture name to a network builder.
Builders import at module level.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `DEFAULT_ARCH` | constant | Kind to default architecture name |
| `default_arch` | function | Default name for a kind |
| `known` | function | Frozen set of kind/name pairs |
| `resolve_arch` | function | Fill an empty name and reject unknown pairs |
| `build` | function | Construct an untrained logits graph |

## Inputs and outputs

`default_arch(kind) -> str`.

`known() -> frozenset[tuple[str, str]]`.

`resolve_arch(kind, arch) -> str`.

`build(kind, arch, in_channels) -> nn.Module`. Raises `ValueError` on an unknown
pair.

## Behavior

1. Empty `arch` selects `resnet50` for the classifier and `unet` for the
   segmentor.
2. `build` calls the matching constructor.

## Errors and faults

`ValueError` on an unknown kind or architecture name.

## Messages

None.

## Configuration

Known pairs: `classifier/resnet50` and `segmentor/unet`.

## Constraints

Adding a family is one `nn.Module` builder plus one registry row.

## Related documents

- [`tools.inference.arch`](../arch.md)
- [`tools.inference.train`](../train.md)
- [`tools.inference.export`](../export.md)
