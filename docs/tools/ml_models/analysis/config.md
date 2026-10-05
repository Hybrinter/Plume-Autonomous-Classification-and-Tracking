# tools.ml_models.analysis.config

**Source:** `packages/tools/src/tools/ml_models/analysis/config.py`
**Kind:** module

## Purpose

This module declares the strict frozen configuration records for the
dataset-analysis, model-analysis, evaluation, and render boundaries, and
the TOML codecs and digests that give each measurement a stable scientific
identity separate from render style.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `PlotConfig` | dataclass | Formats, dpi, figure size, and font size |
| `ScoreConfig` | dataclass | Probability, matching, bin, and threshold-grid settings |
| `CaptureConfig` | dataclass | Retention mode and capture budgets |
| `GeneralizationConfig` | dataclass | Stratification edges and interval budget |
| `EvaluationConfig` | dataclass | Kind, split, batching, and nested score/capture settings |
| `DatasetAnalysisConfig` | dataclass | Dataset path, output, and nested settings |
| `ModelAnalysisConfig` | dataclass | Run, output, checkpoint, final-test, and nested settings |
| `load_dataset_analysis_config` | function | Strict TOML load; `Result` boundary |
| `load_model_analysis_config` | function | Strict TOML load; `Result` boundary |
| `load_plot_config` | function | Strict TOML load; `Result` boundary |
| `write_config` | function | Exclusive nested-TOML write; `Result` boundary |
| `config_digest` | function | SHA-256 of scientific settings, excluding `out` and `plot` |
| `render_digest` | function | SHA-256 of a measurement id plus `PlotConfig` |

## Inputs and outputs

Constructors take typed arguments and raise `ValueError` on violation.
Loaders take a TOML `Path` and return `Result[config, str]`.
`write_config` takes a destination `Path` and a config and returns
`Result[None, str]`. `config_digest` and `render_digest` return lowercase
64-hex strings.

## Behavior

Every record validates at construction: probabilities lie in `[0, 1]`,
bins and batch sizes are positive exact integers, edge lists are finite,
nonnegative, and strictly increasing, and paths are nonblank. Loaders
reject missing files, malformed TOML, unknown fields, and invalid values.
`write_config` serializes nested settings as TOML sections, omits `None`
optionals, and refuses to overwrite an existing file. `config_digest`
changes when inputs, checkpoint, settings, or seed change but not when
`out` or `plot` change; `render_digest` changes with the measurement id
or any render setting.

## Errors and faults

Validation failures raise `ValueError`. I/O and TOML failures return
`Err` strings. Overwrite attempts on `write_config` return `Err`.

## Messages

None.

## Configuration

These records are the configuration. `ModelAnalysisConfig.dataset` is an
optional evaluation-only dataset override and stays `None` unless set.

## Constraints

- All records are frozen slots dataclasses with `extra="forbid"`.
- `formats` are limited to `png`, `svg`, and `pdf`, unique and nonempty.
- `examples_per_family` cannot exceed `max_preview_images`.
- Scientific identity excludes `out` and `plot`; render identity pairs the
  frozen measurement id with `PlotConfig`.
- No scoring, capture, or rendering executes from this module.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.dataset`](dataset.md)
- [`tools.ml_models.analysis.model`](model.md)
