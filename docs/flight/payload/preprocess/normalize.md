# flight.payload.preprocess.normalize

**Source:** `packages/flight/src/flight/payload/preprocess/normalize.py`
**Kind:** pure module

## Purpose

This module scales calibrated band planes from digital numbers to the [0, 1] float32
domain. The model input contract and quality thresholds assume this range. It also
rescales a frame taken at a different exposure/gain to a reference operating point.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `full_scale_dn` | function | ADC code maximum from bit depth or an explicit override |
| `normalize_dn` | function | Divides by ADC full scale and clips to [0, 1] |
| `scale_to_reference_exposure` | function | Rescales planes to a reference exposure/gain |

## Inputs and outputs

`full_scale_dn(bit_depth, adc_max_dn=None)` returns the full scale in DN
(`2**bit_depth - 1` unless `adc_max_dn` is given).

`normalize_dn(planes, bit_depth, adc_max_dn=None)` takes `(C, H, W)` calibrated DN
values and returns `(C, H, W)` float32 in [0, 1].

`scale_to_reference_exposure(planes, exposure_us, gain_db, reference_exposure_us,
reference_gain_db)` applies `signal * (t_ref / t) * 10**((g_ref - g) / 20)` and
returns `(C, H, W)` float32. When any scalar is non-finite or either exposure is
non-positive it returns the input unchanged as a float32 copy.

## Behavior

1. `normalize_dn` computes full scale, divides each element, clips to [0, 1], and
   casts to float32.
2. `scale_to_reference_exposure` multiplies by the exposure ratio and the dB gain
   converted to a linear voltage ratio, for dark-corrected (bias-free) planes.

## Errors and faults

None.

## Messages

None.

## Configuration

Uses `SensorConfig.bit_depth` (default 12, full scale 4095); `adc_max_dn` covers pixel
formats like left-aligned Mono16.

## Constraints

All functions are pure with no I/O. Saturation detection downstream treats values at
1.0 as saturated after clipping. `scale_to_reference_exposure` does not clip;
`normalize_dn` clips afterwards.

## Related documents

- [`flight.payload.preprocess`](preprocess.md)
- [`flight.payload.preprocess.quality`](quality.md)
