# Documentation style

**Source:** `docs/style/`
**Kind:** documentation system

## Purpose

This package holds the writing rules and templates for PACT descriptive documentation
and architecture decision records.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`ste-guide`](style/ste-guide.md) | guide | STE100-inspired house writing rules |
| [`technical-names`](style/technical-names.md) | list | Approved domain nouns and verbs |
| [`module-template`](style/module-template.md) | template | Page layout for one source module |
| [`directory-template`](style/directory-template.md) | template | Page layout for one source directory |
| [`adr-template`](style/adr-template.md) | template | Page layout for one ADR |

## Constraints

- Descriptive pages state the system as it is. They do not explain design choices.
- ADR pages hold design choices. Descriptive pages do not link to ADR pages.
- Coding standards live under `.claude/`, not in this tree.
