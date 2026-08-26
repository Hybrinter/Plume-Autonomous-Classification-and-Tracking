# tools.model.__main__

**Source:** `packages/tools/src/tools/model/__main__.py`
**Kind:** module

## Purpose

This module is the `python -m tools.model` entry. It parses `train`, `export`,
and `accept` subcommands.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `main` | function | Parse argv and dispatch. Returns a process exit code. |

## Inputs and outputs

`main(argv=None) -> int`. `argv` is the argument list without the program name.

## Behavior

1. Parse a required subcommand: `train`, `export`, or `accept`.
2. Print a scaffold message to stderr.
3. Return exit code 2. The CLI is not wired in this layer. Use
   `tools.model.accept.accept_artifact` from Python for acceptance.

## Errors and faults

Argparse raises `SystemExit` on missing or unknown subcommands.

## Messages

None.

## Configuration

None.

## Constraints

This layer does not run training or export. Exit code 2 marks an unimplemented
CLI subcommand.

## Related documents

- [`tools.model`](../model.md)
