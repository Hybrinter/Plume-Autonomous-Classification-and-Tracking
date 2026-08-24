# tools.analysis.report

**Source:** `packages/tools/src/tools/analysis/report.py`
**Kind:** module

## Purpose

Report writes a self-contained static bundle per captured scenario: datasets, figures,
Markdown and HTML summaries, and a deterministic manifest.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SCHEMA_VERSION` | constant | Manifest schema version (1) |
| `RunReport` | class | Paths and manifest for one scenario run |
| `SuiteReport` | class | Suite index plus per-run reports |
| `write_run_report` | function | Emit one run's full bundle |
| `write_suite_report` | function | Emit every run plus suite index |

## Inputs and outputs

**`write_run_report(run, out_dir) -> RunReport`**

- Writes `data/` (long, stats, per-group wide Parquet and CSV), `figures/`, `summary.md`,
  `summary.html`, and `manifest.json`.

**`write_suite_report(runs, suite, out_dir) -> SuiteReport`**

- Writes each run under `out_dir/<scenario>/` plus suite-level `index.md`, `index.html`, and
  `manifest.json`.

## Behavior

1. Call `summarize` on the capture.
2. Write long, stats, and wide frames in both Parquet and CSV.
3. For each plot group, call `build_group_figures` and `save_figures` under `figures/<group>/`.
4. Build manifest with scenario metadata, capture counts, group figure lists, and headline
   outcomes (SAFE latched, final modes, fault totals, storage eviction, downlink peak).
5. Render Markdown and HTML summaries with embedded figure links and per-group stats tables.
6. Suite index links each scenario summary with SAFE-ever and final gimbal state columns.

## Errors and faults

None at library level. File I/O raises normally.

## Messages

None directly. Outcomes derive from captured wide-frame columns.

## Configuration

None.

## Constraints

- No wall-clock timestamps in manifests. Bundles are byte-reproducible.
- Headline outcomes include: `safe_latched_end`, `safe_ever`, `final_gimbal_state`,
  `final_system_mode`, `stow_engaged_ever`, `total_faults`, `final_model_deploy_state`,
  `storage_entries_evicted`, `downlink_pending_peak`, `final_launch_lock_state`.

## Related documents

- [`tools.analysis`](analysis.md)
- [`tools.analysis.stats`](stats.md)
- [`tools.analysis.plots`](plots.md)
- [`tools.analysis.cli`](cli.md)
