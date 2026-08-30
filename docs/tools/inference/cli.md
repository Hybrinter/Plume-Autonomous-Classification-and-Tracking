# tools.inference.cli

**Source:** `packages/tools/src/tools/inference/cli.py`
**Kind:** module

## Purpose

The inference CLI runs training, evaluation, reporting, export, acceptance, and
dataset fetch workflows.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `app` | Typer application | Inference command group |
| `main` | function | Invoke the command group and return an exit code |
| `InferenceKind` | enum | Classifier and segmentor command choices |

## Inputs and outputs

`main(argv=None) -> int` accepts an optional argument vector without the program
name. Commands print generated paths, tables, acceptance details, or dataset
status.

## Behavior

1. `train` overlays options on `TrainConfig` and writes a run directory. Flags
   include `--loss`, `--focal-gamma`, `--focal-alpha`, `--amp`, `--patience`,
   and `--eval-interval`.
2. `eval` scores a checkpoint on a named split. The default split is `val`.
3. `report` writes figures and `report.md` into a run directory.
4. `list` prints local run summaries. `compare` prints a side-by-side table.
   `rank` prints the same columns in val-metric order. `pareto` prints the
   size-versus-quality frontier from the run catalog. Every point is read from
   one split so the comparison is like for like.
5. `sweep` expands a space TOML, trains each trial, scores val, and writes
   JSONL. `arches` prints the curated catalog of kind/name pairs.
6. `export` writes an ONNX graph and JSON manifest. `--int8` also writes a
   sibling INT8 QDQ pair.
7. `accept` runs the classifier or segmentor intake gate and optionally promotes
   an accepted artifact. `--scenes-dir` supplies a processed pack for golden
   scenes. `--scenes-split` names the split (default `test`). `--scenes-limit`
   caps the scene count; zero takes the whole split. Without `--scenes-dir` the
   quality check has no scenes and the gate fails that check.
8. `finalize` scores the test split of a run, exports FP32 and INT8, and runs
   the golden-scene gate. `--promote` copies the preferred accepted artifact.
9. `fetch` delegates dataset status, download, unpack, and labeled preprocess
   work to `tools.inference.fetch`.

## Errors and faults

Invalid command input returns the Click usage-error exit code. A failed train
(`ValueError` or `FileExistsError`) or a failed acceptance gate returns 1.
Missing optional export dependencies print the import error and return 1.

## Messages

None.

## Configuration

Commands expose the same train, export, accept, eval, and fetch options as their
library configuration objects and functions. `train` overlays every
`TrainConfig` field. `--overwrite` sets `overwrite` true. `--shuffle` and
`--augment` set those flags true. `--amp` sets `amp` true. The `pareto`
command adds `--metric`, `--cost` (`n_params` or `flops`), `--kind`,
`--split` (`val` or `test`, default `val`), `--from-jsonl`, `--by-arch`,
`--baseline`, `--spread`, `--auto-spread`, `--neighbors`, and `--write-space`.
`--from-jsonl` keeps identifiers recorded in that sweep file. The flag may be
passed more than once; the identifier sets are joined. `--by-arch`
averages extra seeds of the same architecture. `--baseline` prints the cheapest
frontier point that holds that published score, plus `--neighbors` (default 1)
on each side. `--spread` (default 0) is the allowed drop below the baseline.
`--auto-spread` raises that drop to the seed range of the first-pass knee
architecture. `--write-space` replaces `PLACEHOLDER_SET_FROM_STAGE_2` in that
space TOML. The flag may be passed more than once. `--write-space` requires
`--baseline`. The `accept` command adds
`--scenes-dir`, `--scenes-split`, and `--scenes-limit` for golden scene
loading.

## Constraints

- Command callbacks remain thin wrappers around importable library functions.
- Torch and ONNX imports stay behind their optional extras.

## Related documents

- [`tools.inference`](../inference.md)
- [`tools.inference.train`](train.md)
- [`tools.inference.eval`](eval.md)
- [`tools.inference.report`](report.md)
- [`tools.inference.runs`](runs.md)
- [`tools.inference.pareto`](pareto.md)
- [`tools.inference.sweep`](sweep.md)
- [`tools.inference.export`](export.md)
- [`tools.inference.accept`](accept.md)
- [`tools.inference.finalize`](finalize.md)
- [`tools.inference.fetch`](fetch.md)
