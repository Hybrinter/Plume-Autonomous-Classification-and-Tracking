# PACT documentation

## Start here

| Document | Role |
| --- | --- |
| [`style.md`](style.md) | Writing rules and templates |
| [`flight.md`](flight.md) | Flight software package |
| [`sim.md`](sim.md) | Simulation and SIL package |
| [`gse.md`](gse.md) | Ground support equipment package |
| [`tools.md`](tools.md) | Analysis and acceptance tooling |
| [`adr.md`](adr.md) | Repository-scope decision index |

Design and performance studies live under [`analysis/`](../analysis/) (`pact-analysis`). They
are not STE-mirrored. `tools.analysis` is SIL capture; `analysis.*` is design studies.

## Document kinds

1. **Descriptive pages** under `docs/flight`, `docs/sim`, `docs/gse`, and
   `docs/tools` mirror the source trees. They state what the code does now.
2. **ADR pages** under `docs/adr` (repository scope) and
   `docs/<package>/adr` (package scope) record why a decision was made.
3. **Requirements** under `docs/requirements` hold the VCRM.
4. **Validation** under `docs/validation` holds PIL and HIL procedures.

## Mirror rule

For each source directory `X/`, the docs tree has both `X.md` and `X/`.
For each source module `foo.py` (except empty `__init__.py`), the docs tree
has `foo.md`.

## Reference rules

- Descriptive pages may link to other descriptive pages.
- Descriptive pages must not link to ADR pages or cite ADR identifiers.
- ADR pages may cite other ADR pages.
- Coding standards live in `.claude/rules/`.

## Writing style

Follow [`style/ste-guide.md`](style/ste-guide.md).
