# tools.ml_models.export.pair

**Source:** `packages/tools/src/tools/ml_models/export/pair.py`
**Kind:** module

## Purpose

This module decides whether a training run matches the flight input contract
and writes the classifier plus segmentor pair JSON.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `flight_promotable` | function | True when a run matches `InferenceConfig` |
| `write_pair_manifest` | function | Write the deploy pair JSON |

## Inputs and outputs

`flight_promotable(run_dir, inference=None) -> bool`.

`write_pair_manifest(classifier_sidecar, segmentor_sidecar, dest) -> dict`.
The file holds `version`, `classifier.input_shape`, `classifier.output_shape`,
`segmentor.input_shape`, and `segmentor.output_shape`.

## Behavior

1. Read `config.toml` and `summary.json` from the run directory.
2. Read `in_channels` and `band_names` from the summary, then the config, then
   `checkpoints/best.pt` or `checkpoints/last.pt`.
3. Take spatial size from summary `input_height_px` and `input_width_px` when
   both are present. Otherwise use summary `frame_hw`, then config
   `[canvas] frame_hw`, then the config crop size.
4. Return true when the channel count equals `len(input_bands)`, the band
   names equal `input_bands`, and the spatial size equals `input_height_px`
   by `input_width_px`. With today's defaults that is 3,
   `("BLUE", "GREEN", "RED")`, 1544, and 2064.
5. Ignore `radiometry` and `ingest_path` for the boolean result.
6. `write_pair_manifest` requires both runs to be flight-promotable. It
   requires both sidecars to match the flight input `(1, 3, 1544, 2064)`,
   classifier output `(1, 1)`, and segmentor output `(1, 1, 1544, 2064)`.
   It writes that object to `dest`.

## Errors and faults

`write_pair_manifest` raises `ValueError` when a run is missing, a run is not
flight-promotable, the sidecar versions differ, or a sidecar shape does not
match the flight contract. `load_manifest` raises on a malformed sidecar.

## Messages

None.

## Configuration

`inference` defaults to `InferenceConfig()`. The pair blob uses that same
default contract. The check does not require `norm` equal to `normalize_dn`.

## Constraints

The module does not import `flight.core` or `tools.analysis`. It does not
construct `ModelDeployService`. A 76 px run, a 512 px run, or a 4-channel run
is not flight-promotable.

## Related documents

- [`tools.ml_models.export`](../export.md)
- [`tools.ml_models.export.accept`](accept.md)
- [`tools.ml_models.cli`](../cli.md)
- [`flight.libs.config`](../../../flight/libs/config.md)
