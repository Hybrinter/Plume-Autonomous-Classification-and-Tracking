# ADR-TOOLS-0005: Use a nested tools CLI and inference package

**Status:** Accepted
**Date:** 2026-08-27
**Topic:** restructure
**Supersedes:** ADR-TOOLS-0001
**Superseded-by:** none
**Related:** ADR-REPO-0004, ADR-REPO-0011, ADR-TOOLS-0002, ADR-TOOLS-0003, ADR-TOOLS-0004

## Context

`pact-tools` contains model-development workflows and whole-system SIL
observability. More engineering utilities will be added for individual payload
concerns such as inference and control. A single `tools.payload` package would
mix unrelated workflows, while separate top-level commands would make the
installed interface harder to discover.

The model package owns training, export, and acceptance of the classifier and
segmentor used by flight inference. Its `tools.model` name describes its
artifacts instead of the payload concern that consumes them. The existing
`python -m` entry points also use separate argparse parsers and do not provide
one installed command.

## Decision

Organize payload engineering utilities as sibling packages named for payload
concerns. Rename `tools.model` to `tools.inference`. Do not add a
`tools.payload` umbrella. Future concerns, such as controller development, may
add sibling packages when their workflows exist.

Use Typer for nested commands. Install `pact-tools` as the named console
command. Keep `python -m tools` and `python -m tools.<package>` as aliases.
Each tools package owns its Typer application, and the root CLI registers
applications explicitly.

The inference CLI owns train, export, accept, and dataset-fetch commands.
Remove the `tools.model` and `tools.accept` compatibility paths.

Keep `tools.analysis` as a sibling package for passive whole-system SIL
observability. Its CLI remains a thin wrapper because its workflows can change
as flight software matures.

This decision does not define CLIs for GSE, simulation, or flight software.

## Consequences

- Inference workflows use `pact-tools inference ...`,
  `python -m tools inference ...`, or `python -m tools.inference ...`.
- Analysis workflows use `pact-tools analysis ...`,
  `python -m tools analysis ...`, or `python -m tools.analysis ...`.
- Library users import `tools.inference`; old `tools.model` and `tools.accept`
  imports fail.
- New tools concerns add one package-owned Typer application and one explicit
  root registration.
- Typer and Click become tools-package dependencies. They do not enter the
  flight image.
- The `train` and `export` extras retain their names because they identify
  dependency roles, not Python packages.

## Alternatives considered

- Keep `tools.model` — does not align the package with the flight inference
  concern.
- Add `tools.payload` — creates an umbrella with unrelated submodule
  workflows.
- Keep independent argparse entry points — provides no discoverable installed
  command as the tools package grows.
- Add a dynamic plugin registry — adds indirection without a current extension
  requirement.
