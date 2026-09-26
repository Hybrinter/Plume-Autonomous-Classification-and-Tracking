# flight.payload.preprocess.band_select

**Source:** `packages/flight/src/flight/payload/preprocess/band_select.py`
**Kind:** pure module

## Purpose

Band order lives in `flight.libs.types.BAND_ORDER`. This module exposes that
order as strings. Preprocess does not reorder channels.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `canonical_band_names` | function | Returns `("BLUE", "GREEN", "RED")` |

## Inputs and outputs

`canonical_band_names()` takes no arguments. It returns one string per
`BAND_ORDER` entry.

## Behavior

1. Read `BAND_ORDER`.
2. Return each member's value, in that order.

## Errors and faults

None.

## Messages

None.

## Configuration

None.

## Constraints

The driver stacks planes in `BAND_ORDER`. The model reads the same tuple.
This module does not gather or permute arrays.

## Related documents

- [`flight.payload.preprocess`](../preprocess.md)
- [`flight.libs.types.enums`](../../libs/types/enums.md)
