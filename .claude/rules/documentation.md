# Documentation rules for agents

## Split of responsibility

- **Descriptive docs** (`docs/flight`, `docs/sim`, `docs/gse`, `docs/tools`)
  state the system as it is. Use the STE100-inspired guide in
  `docs/style/ste-guide.md`.
- **ADRs** (`docs/adr`, `docs/<package>/adr`) record design decisions.
- **Design briefs** (`docs/design`) specify architecture that is not yet
  as-built. They may use equations and future-tense language. They are not
  STE-mirrored and must not cite architecture-decision identifiers.
- **`.claude/rules/`** holds coding standards and agent invariants.

Do not put design rationale in descriptive docs. Do not put coding-style
preferences in ADRs. Do not put an as-yet-unbuilt controller in a descriptive
page; put it in `docs/design` until the code exists.

## Mirror maintenance

When you add, rename, or delete a source module or package directory under
`packages/{flight,sim,gse,tools}/src/`, update the matching descriptive page
in the same change. Empty `__init__.py` files fold into the parent directory
page and do not get their own markdown file. `packages/analysis/` is not
STE-mirrored.

## ADR rules

- One decision per file under `adr/`.
- Never rewrite an accepted ADR body.
- You may change only the `Status` and `Superseded-by` fields on an old ADR
  when a new ADR supersedes it.
- New package decisions use IDs `ADR-FLIGHT-NNNN`, `ADR-SIM-NNNN`,
  `ADR-GSE-NNNN`, or `ADR-TOOLS-NNNN`.
- Cross-package decisions use `ADR-REPO-NNNN` under `docs/adr/`.

## Forbidden ADR references

Do not cite ADR identifiers or ADR paths in:

- descriptive docs
- source comments
- docstrings
- tests
- `CLAUDE.md` prose that is not an ADR index

Only ADR files and ADR index pages may cite ADRs.

## Stub docs

If the source unit is a stub or empty scaffold, the descriptive page must
say so in Purpose and Constraints. Do not invent behavior that the code
does not implement.
