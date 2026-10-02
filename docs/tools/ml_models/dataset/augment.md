# tools.ml_models.dataset.augment

**Source:** `packages/tools/src/tools/ml_models/dataset/augment.py`
**Kind:** module

## Purpose

This module applies offline dihedral augmentation to dataset rows. Val and
test rows keep the identity element; train rows expand over the legal
element list.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ELEMENT_NAMES` | constant | The eight dihedral names |
| `SHAPE_PRESERVING` | constant | The four elements that keep H and W |
| `AugmentRecipe` | class | Element names requested for train rows |
| `legal_elements` | function | Elements legal for a tile size |
| `apply_dihedral` | function | Apply one element to a `(C, H, W)` array |

## Inputs and outputs

`AugmentRecipe(elements=ELEMENT_NAMES)` is a frozen pydantic dataclass.
`elements` must be a non-empty, unique subset of `ELEMENT_NAMES`.

`legal_elements(height, width) -> tuple[str, ...]` returns all eight names
on a square tile and `SHAPE_PRESERVING` otherwise.

`apply_dihedral(image, element) -> np.ndarray` takes `(C, H, W)` and
returns a contiguous `(C, H', W')` copy.

## Behavior

1. Each element is `k` quarter-turns after an optional left-right flip:
   `id`, `rot90`, `rot180`, `rot270`, `flip_h`, `flip_v`, `transpose`,
   `anti_transpose`.
2. A 90 degree turn or a transpose swaps H and W. A non-square tile drops
   those elements and keeps along-track compression on H.
3. GSD is not an input to this module. The build filters the recipe against
   the stored GSD itself.

## Errors and faults

`ValueError` when the recipe is empty, names an unknown or duplicate
element, when `legal_elements` gets a size below 1, or when
`apply_dihedral` gets an unknown element or a non-3-D array.

## Messages

None.

## Configuration

The default recipe is all eight elements. `BuildSpec.augment` overrides
the list. There is no TOML file in this module.

## Constraints

Elements apply to `(C, H, W)` arrays, images and masks alike. This module
does not import torch.

## Related documents

- [`tools.ml_models.dataset`](../dataset.md)
- [`tools.ml_models.dataset.build`](build.md)
- [`tools.ml_models.dataset.spec`](spec.md)
