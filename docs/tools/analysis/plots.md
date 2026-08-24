# tools.analysis.plots

**Source:** `packages/tools/src/tools/analysis/plots/`
**Kind:** package

## Purpose

The plots package builds matplotlib figures for each signal group. Report emission calls one
builder per group against that group's wide frame.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`common`](plots/common.md) | module | Shared headless figure primitives |
| [`system`](plots/system.md) | module | System rollup figures |
| [`bus`](plots/bus.md) | module | Message bus throughput and backlog |
| [`payload`](plots/payload.md) | module | Payload FSM, pointing, tracking, output |
| [`fault`](plots/fault.md) | module | FDIR latch, watchdog, fault codes |
| [`iss_iface`](plots/iss_iface.md) | module | Link state, ingress, upload |
| [`thermal`](plots/thermal.md) | module | Temperature vs limit |
| [`electrical`](plots/electrical.md) | module | Power vs limit |
| [`command_router`](plots/command_router.md) | module | Hazardous gate and routing |
| [`storage`](plots/storage.md) | module | Quota usage and eviction |
| [`downlink`](plots/downlink.md) | module | Queue, AOS, priority backlog |
| [`mechanical`](plots/mechanical.md) | module | Launch-lock state and activity |
| [`model_deploy`](plots/model_deploy.md) | module | Model lifecycle state |

## Package interface

| Name | Kind | Description |
| --- | --- | --- |
| `GroupPlots` | class | Group name plus figure builder callable |
| `PLOT_GROUPS` | constant | Ordered group registry |
| `build_group_figures` | function | Render one group's figures from its wide frame |

## Interactions

Plot builders import `tools.analysis.plots.common` only. They read pandas wide frames produced
by the recorder. They do not drive the SIL.

## Constraints

- Agg backend is selected on import in `common`. No display is required.
- Builders drop panels when columns are absent or carry no finite data.
- Figure order follows `PLOT_GROUPS`, mirroring report emission order.

## Related documents

- [`tools.analysis`](analysis.md)
- [`tools.analysis.report`](report.md)
- [`tools.analysis.plots.common`](plots/common.md)
