# tools.ml_models.dataset.sources.zenodo.bandmap

**Source:** `packages/tools/src/tools/ml_models/dataset/sources/zenodo/bandmap.py`
**Kind:** module

## Purpose

This module computes Sentinel-2 band weights for a camera channel from
modeled spectral responses. It writes VIS weights as a prism weight table
and reports SWIR weights for the candidate band edges.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SpectralResponse` | Protocol | Relative response at given wavelengths |
| `SuperGaussianBand` | dataclass | Band shape from center, FWHM, and order |
| `LogisticPassband` | dataclass | Smooth top-hat with half-power edges |
| `ProductResponse` | dataclass | Pointwise product of responses |
| `S2A_BANDS` | constant | Sentinel-2A center and FWHM per band, metres |
| `DEFAULT_BASIS` | constant | Interpolation bands: B1-B7, B8A, B11, B12 |
| `ChannelMapping` | dataclass | Weights, band capture, coverage, centroid |
| `SensitivityCase` | dataclass | One SWIR edge and basis case |
| `wavelength_grid` | function | 350-2500 nm grid in 0.5 nm steps, metres |
| `solar_blackbody` | function | Relative 5772 K Planck spectrum |
| `s2_band` | function | Order-3 super-Gaussian for one band |
| `imx267_placeholder_channels` | function | Placeholder blue, green, red behind the VIS coating |
| `hd5_placeholder_channel` | function | Placeholder HD5 passband |
| `map_channel` | function | Weights and diagnostics for one channel |
| `vis_weight_table_toml` | function | VIS mappings as prism TOML text |
| `swir_edge_sensitivity` | function | HD5 mappings for two edges and two bases |

## Inputs and outputs

All wavelengths are metres. `map_channel(name, channel, basis,
solar_weighted)` returns a `ChannelMapping`:

- `weights`: basis band id to weight. The weights sum to 1.
- `band_capture`: every Sentinel-2 band id to the fraction of that band's
  solar-weighted response that the channel passes.
- `coverage`: fraction of the channel signal inside any basis band shape.
- `centroid_m`: solar-weighted response centroid.

`vis_weight_table_toml(table_id, mappings)` returns TOML text that
`prism.load_weight_table` accepts.

## Behavior

1. The channel signal is the response times a 5772 K blackbody, unless
   `solar_weighted` is false. Atmospheric absorption is not modeled.
2. Surface reflectance between basis band centers is linear in wavelength.
   Beyond the end centers it is constant. The weight of basis band `b` is
   the integral of the signal times the tent function of `b`, divided by
   the integral of the signal. The weights of one channel sum to 1.
3. Each Sentinel-2 band is an order-3 super-Gaussian at the Sentinel-2A
   center and FWHM.
4. B10 is never a default basis band. It is L1C top-of-atmosphere data;
   the other bands are L2A surface data. B8 and B9 are also out of
   the default basis.
5. The placeholder IMX267 channels are Gaussians (blue 465 nm / 100 nm
   FWHM, green 535 / 100, red 610 / 90) times a 425-675 nm logistic coating
   with a 5 nm edge scale. Lens transmission is flat.
6. The placeholder HD5 channel is a logistic passband with a 10 nm edge
   scale. `swir_edge_sensitivity` runs 1.0-1.65 um and 0.9-1.7 um, each
   with the default basis and with B9 added.
7. `vis_weight_table_toml` drops weights below `1e-4`, renormalizes,
   rounds to 6 places, and gives the rounding residual to the largest
   weight.

## Errors and faults

`ValueError` on a non-positive FWHM, order, or peak; an inverted passband or
a non-positive edge scale; an empty product; an empty, repeated, or unknown
basis; a channel with no response on the grid; a missing color; or an empty
table id. `KeyError` from `s2_band` on an unknown band id.

## Messages

None.

## Configuration

None. The `dataset band-weights` CLI command writes the VIS table and a JSON
report.

## Constraints

The camera curves are placeholders, not datasheet curves. Table ids from
this module carry `placeholder`. This module does not import torch or
rasterio.

## Related documents

- [`tools.ml_models.dataset.sources.zenodo`](../zenodo.md)
- [`tools.ml_models.dataset.sources.zenodo.prism`](prism.md)
- [`tools.ml_models.dataset.sources.zenodo.bands`](bands.md)
- [`tools.ml_models.cli`](../../../cli.md)
