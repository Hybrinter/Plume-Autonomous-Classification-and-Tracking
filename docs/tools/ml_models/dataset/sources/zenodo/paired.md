# tools.ml_models.dataset.sources.zenodo.paired

**Source:** `packages/tools/src/tools/ml_models/dataset/sources/zenodo/paired.py`
**Kind:** module

## Purpose

This module renders each Zenodo 4250706 scene as BLUE, GREEN, RED, and SWIR
planes for the candidate VIS and SWIR cameras, at one GSD bin per gimbal
elevation. `ZenodoPairedSource` is the `RawSource` for the paired build.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `PAIRED_BANDS` | constant | `("BLUE", "GREEN", "RED", "SWIR")` |
| `PairedConfig` | dataclass | Camera, blur, noise, and SWIR-mix settings |
| `load_paired_config` | function | Read a `PairedConfig` TOML |
| `CameraBin` | dataclass | VIS `BinSpec` and the SWIR GSD at one elevation |
| `camera_bins` | function | One bin per elevation from flight footprint geometry |
| `resize_bilinear` | function | Center-aligned resize of `(C, h, w)` planes |
| `gaussian_blur` | function | Separable Gaussian blur of `(C, H, W)` planes |
| `sharpen_b1` | function | Add B2 detail finer than 60 m to B1 |
| `add_sensor_noise` | function | Shot, read, and quantization noise |
| `scene_mix` | function | Per-scene SWIR mix in `[0, 1)` |
| `swir_weights_for` | function | SWIR weights and mix for one scene |
| `render_scene` | function | One stack to `(4, H, W)` unit reflectance |
| `ZenodoPairedSource` | class | Paired `RawSource` |

## Inputs and outputs

`render_scene(stack_dn, band_ids, vis_table, cfg, camera_bin,
swir_weights, rng=None)` takes a native `(C, 120, 120)` L2A stack and
returns float32 `(4, H, W)` in `[0, 1]`, where `(H, W)` is `bin_hw` of the
VIS bin. `ZenodoPairedSource(images_tar, labels_tar, vis_table, cfg)` emits
one row per scene and bin. Row ids are `<stem>-elNN`.

## Behavior

1. `camera_bins` calls `pixel_gsd` at the frame center for each elevation,
   for a nominal circular orbit at `altitude_m`, evaluated at
   `plume_height_m`. The VIS camera is the band plane: 1920 x 1080 samples
   at 6.9 um behind 100 mm. The SWIR camera is 1280 x 1024 at 5 um behind
   40 mm. The default elevations are 5, 15, 25, 35, and 45 deg.
2. Row GSD is metres per VIS band sample. Rows carry `theta_g_deg` equal to
   the bin elevation and `gsd_nominal` true.
3. With `b1_sharpen`, B1 becomes `B1 + B2 - lowpass(B2)`, where `lowpass`
   area-averages B2 to 60 m and resizes it back with `resize_bilinear`.
4. VIS planes are `mix_prism` of the stack with the VIS table. The SWIR
   plane is the weighted sum of clipped L2A reflectance.
5. Each camera blurs on the native 10 m grid with a Gaussian of sigma
   `0.42 lambda N` at the focal plane, combined in quadrature with
   `extra_blur_px`, then area-samples to its own GSD. The SWIR plane is
   then resized onto the VIS grid with `resize_bilinear`.
6. With `noise`, each camera adds noise at its own grid before the SWIR
   resize: electrons `rho / saturation_reflectance * full_well`, Gaussian
   sigma `sqrt(e + read_noise**2)`, then rounding to `bit_depth` counts.
7. With `swir_randomize`, the SWIR weights blend `swir_weights_low` and
   `swir_weights_high` by a mix drawn from a CRC32 seed of the scene stem
   and `seed`. All bins of a scene share the mix. Rows carry the condition
   `swir_mix` with two decimals. Without it, every scene uses
   `swir_weights`, which default to the `bandmap` HD5 datasheet-edge
   weights.
8. Labels, split groups, and masks follow `ZenodoSource`; masks rasterize
   on the VIS grid.

## Errors and faults

`ValueError` on a non-positive geometry, optics, or sensor value; empty or
repeated elevations; an elevation that misses the Earth; a bin finer than
10 m; an unknown or wrongly typed TOML key; weights without a positive
total; a SWIR weight on a missing band; or noise without a generator.
`bin_hw` and the archive readers raise as in `ZenodoSource`.

## Messages

None.

## Configuration

`PairedConfig` defaults: id `vis_swir_paired_estimate_v0`, 440 km, 2 km
plume height, VIS f/11 at 550 nm, SWIR f/2.8 at 1300 nm, B1 sharpening on,
SWIR randomization off, noise off. Sensor noise values are estimates: VIS
full well 10500 e and read noise 2.5 e, SWIR full well 30000 e and read
noise 40 e, 12 bits. The `--paired-config` CLI option reads a TOML of these
field names.

## Constraints

Camera values are estimates, not datasheet curves or measurements. A row
covers one 1.2 km Zenodo scene, a small part of one flight tile. There is
no atmospheric model and no dark current. Randomization and noise apply
to every split. This module does not import torch.

## Related documents

- [`tools.ml_models.dataset.sources.zenodo`](../zenodo.md)
- [`tools.ml_models.dataset.sources.zenodo.adapt`](adapt.md)
- [`tools.ml_models.dataset.sources.zenodo.bandmap`](bandmap.md)
- [`tools.ml_models.dataset.sources.zenodo.prism`](prism.md)
- [`tools.ml_models.dataset.build`](../../build.md)
