# tools.ml_models.data.norm

**Source:** `packages/tools/src/tools/ml_models/data/norm.py`
**Kind:** module

## Purpose

This module scales band planes with one of three recipes: ADC full scale, unit
clip, or a per-band z-score.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `BandStats` | class | Per-channel mean and population standard deviation |
| `apply_normalize_dn` | function | Call flight `normalize_dn` |
| `apply_unit` | function | Clip to `[0, 1]` float32 |
| `fit_band_stats` | function | Fit moments on `(C, H, W)` or `(N, C, H, W)` |
| `apply_band_z` | function | Subtract the mean and divide by the std |
| `apply_norm` | function | Dispatch on the recipe name |

## Inputs and outputs

`apply_normalize_dn(planes, bit_depth) -> np.ndarray`.

`apply_unit(planes) -> np.ndarray`.

`fit_band_stats(stack) -> BandStats`. `mean` and `std` are tuples of length C.

`apply_band_z(planes, stats) -> np.ndarray`.

`apply_norm(planes, norm, *, bit_depth=12, stats=None) -> np.ndarray`.

Output dtype is float32.

## Behavior

1. `apply_normalize_dn` calls `normalize_dn`. Full scale is
   `2**bit_depth - 1`. That value maps to 1. `bit_depth` is an int >= 1.
2. `apply_unit` clips every element to `[0, 1]`.
3. `fit_band_stats` accepts `(C, H, W)` or `(N, C, H, W)`. Mean and population
   standard deviation are per channel. Population std uses divisor P, the
   pixel count. Every pixel must be finite. Each returned mean is finite.
   Std is floored at `1e-6` only when the raw std is finite.
4. `apply_band_z` returns `(planes - mean) / std`. Each mean must be finite.
   Each std must be finite and greater than 0.
5. `apply_norm` selects `normalize_dn`, `unit`, or `band_z`. `band_z` without
   `stats` raises `ValueError`.

## Errors and faults

`ValueError` when `bit_depth` is below 1, the stack rank is not 3 or 4, an
axis is empty, a pixel is non-finite, moment length disagrees with the channel
count, a mean is non-finite, a std is non-finite or not positive, or `band_z`
is missing `stats`.

## Messages

None.

## Configuration

`apply_norm` uses bit depth 12 when the caller omits `bit_depth`. The std
floor is `1e-6`. There is no TOML file.

## Constraints

Output dtype is float32. This module does not import torch. It does not import
`flight.payload.inference`, `flight.core`, or `tools.analysis`.

## Related documents

- [`tools.ml_models.data`](../data.md)
- [`tools.ml_models.data.meta`](meta.md)
- [`flight.payload.preprocess.normalize`](../../../flight/payload/preprocess/normalize.md)
