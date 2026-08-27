# Flight package ADRs

**Scope:** `ADR-FLIGHT`
**Directory:** [`adr/`](adr/)

Package-local decisions for `packages/flight`. New records use files under
`adr/` named `NNNN-short-title.md` with title `# ADR-FLIGHT-NNNN: ...`.

## Predecessor records

These repository-scope decisions apply to flight. They stay under `docs/adr/`.

| ID | Decision |
| --- | --- |
| ADR-REPO-0003 | Subsystem-app model over a typed message bus |
| ADR-REPO-0004 | ONNX frozen-artifact detector behind a swappable backend |
| ADR-REPO-0005 | Pure-core + thin-shell apps; `Result` over exceptions |
| ADR-REPO-0006 | ISS-attached reliability posture |
| ADR-REPO-0007 | Raw-mosaic sensor ingest contract |
| ADR-REPO-0008 | Closed-loop gimbal pointing |
| ADR-REPO-0009 | ISS link transport + authenticated command ingress |

## Records

| ID | File | Decision | Status |
| --- | --- | --- | --- |
| ADR-FLIGHT-0001 | [0001-classifier-segmentor-backends.md](adr/0001-classifier-segmentor-backends.md) | ClassifierBackend, SegmentorBackend, and blob DetectorBackend | Accepted |
