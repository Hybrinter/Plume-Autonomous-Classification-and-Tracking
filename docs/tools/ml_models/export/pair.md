# tools.ml_models.export.pair

**Source:** `packages/tools/src/tools/ml_models/export/pair.py`
**Kind:** module

## Purpose

This module decides whether a training run matches the flight input contract
and writes the classifier plus segmentor pair JSON.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `flight_promotable` | function | True when a run matches the flight tile |
| `write_pair_manifest` | function | Write the deploy pair JSON |

## Inputs and outputs

`flight_promotable(run_dir, inference=None) -> bool`.

`write_pair_manifest(classifier_sidecar, segmentor_sidecar, dest) -> dict`.
The file holds `version`, `grid`, `frame_hw`, and both network shapes.
JSON `null` is the dynamic batch axis.

## Behavior

1. Read `config.toml` and `summary.json` from the run directory.
2. Read `in_channels`, `band_names`, `kind`, and `arch` from the summary, then
   the config, then `checkpoints/best.pt` or `checkpoints/last.pt`. An empty
   string is missing.
3. Read `tile_hw` from the summary, then the checkpoint. The sensor frame and
   the config crop are not the graph size.
4. Return true when the channel count equals `len(input_bands)` and the band
   names equal `input_bands`. With today's defaults that is 3 and
   `("BLUE", "GREEN", "RED")`.
5. `tile_hw` must be `(193, 258)`. A classifier arch must be `pactnet`. A
   segmentor arch must be `dilatenet`.
6. Ignore `radiometry` and `ingest_path` for the boolean result.
7. `write_pair_manifest` requires both runs to be flight-promotable. Sidecar
   versions must match. Classifier input is `[null, 3, 193, 258]` and output
   is `[null, 1]`. Segmentor input matches. Segmentor output is
   `[null, 1, 193, 258]`. The JSON also records `grid` `[8, 8]` and
   `frame_hw` `[1544, 2064]`.

## Errors and faults

`write_pair_manifest` raises `ValueError` when a run is missing, a run is not
flight-promotable, the sidecar versions differ, or a sidecar shape does not
match the flight contract. `load_manifest` raises on a malformed sidecar.

## Messages

None.

## Configuration

`inference` defaults to `InferenceConfig()`. Channel count and band names come
from `input_bands`. The graph size is the flight tile. The check does not
require `norm` equal to `normalize_dn`.

## Constraints

The module does not import `flight.core` or `tools.analysis`. It does not
construct `ModelDeployService`. A 76 px run, a 512 px run, and a 1544 by 2064
run are not flight-promotable. A ShuffleNet run or a U-Net run at 193 by 258
is not flight-promotable. A 4-channel run is not flight-promotable.

## Related documents

- [`tools.ml_models.export`](../export.md)
- [`tools.ml_models.export.accept`](accept.md)
- [`tools.ml_models.cli`](../cli.md)
- [`flight.libs.config`](../../../flight/libs/config.md)
