# tools.inference.arch.grammar

**Source:** `packages/tools/src/tools/inference/arch/grammar.py`
**Kind:** module

## Purpose

This module parses underscore-separated architecture-name modifiers. Families
share one token walker. Each family names the tokens it accepts.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ModifierFlags` | class | Accepted tokens and numeric-token labels |
| `NameModifiers` | class | Parsed width, depth, stride, decoder width, and flags |
| `positive_int` | function | Parse a `<prefix><digits>` token |
| `parse_modifiers` | function | Walk tokens under one family's flags |

## Inputs and outputs

`positive_int(token, prefix, label) -> int`. Raises `ValueError` when the
remainder is empty, non-numeric, or below one.

`parse_modifiers(tokens, flags, unknown) -> NameModifiers`. Raises `ValueError`
when a token is outside `flags` or a numeric token is malformed. `unknown` is
the phrase in the unknown-token error, such as `pactnet modifier`.

## Behavior

1. Numeric tokens are `w<N>` (width), `d<N>` (depth), `s<N>` (stride), and
   `x<N>` (decoder width).
2. Flag tokens are `sep`, `full`, and `pt`.
3. `parse_modifiers` honours only the flags that the family sets.
4. `sep` writes `separable=True`. `full` writes `separable=False`. `pt` writes
   `pretrained=True`.
5. Absent numeric tokens stay `None`. Absent `separable` stays `None`. Absent
   `pt` leaves `pretrained` false.
6. Later tokens of the same kind overwrite earlier ones.

## Errors and faults

`ValueError` on a malformed numeric token or an unknown modifier. The unknown
error uses the family's `unknown` phrase.

## Messages

None.

## Configuration

None.

## Constraints

This module has no torch import. Compact, dilated, scratch U-Net, and runet
parsers call `parse_modifiers`.

## Related documents

- [`tools.inference.arch`](../arch.md)
- [`tools.inference.arch.compact`](compact.md)
- [`tools.inference.arch.dilated`](dilated.md)
- [`tools.inference.arch.unet`](unet.md)
- [`tools.inference.arch.registry`](registry.md)
