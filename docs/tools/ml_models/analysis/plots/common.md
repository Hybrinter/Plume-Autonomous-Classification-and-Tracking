# tools.ml_models.analysis.plots.common

**Source:** `packages/tools/src/tools/ml_models/analysis/plots/common.py`
**Kind:** module
**Status:** stub

## Purpose

This module holds figure/export conventions and the render boundary for
frozen evidence directories. It is unavailable until the plotting phase
lands.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `render_analysis` | function | Render boundary over a frozen evidence directory |

## Inputs and outputs

`render_analysis(evidence_dir: Path, cfg: PlotConfig, out: Path) ->
Result[Path, str]`.

## Behavior

The function currently returns `Err` with an explicit unavailable message
and creates no output directory. The `render` CLI command calls it.

## Errors and faults

Always `Err` while unimplemented.

## Messages

None.

## Configuration

`PlotConfig`; see
[`tools.ml_models.analysis.config`](../config.md).

## Constraints

- Rendering consumes frozen evidence only and never reruns inference.
- No figure directory is written while the boundary is unavailable.

## Related documents

- [`tools.ml_models.analysis.plots`](../plots.md)
- [`tools.ml_models.cli`](../cli.md)
