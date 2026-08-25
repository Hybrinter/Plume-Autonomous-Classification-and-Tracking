# Repository ADRs

**Scope:** `ADR-REPO`
**Directory:** [`adr/`](adr/)

This index lists cross-package architecture decisions. Decision bodies under
`docs/adr/` are frozen. Do not rewrite an accepted body. You may change only
the `Status` and `Superseded-by` fields when a later record supersedes one.

New cross-package decisions continue this sequence as `ADR-REPO-NNNN`.
Package-local decisions live under `docs/<package>/adr/`.

## Records

| ID | File | Decision | Status |
| --- | --- | --- | --- |
| ADR-REPO-0001 | [0001-python-only-drop-rust.md](adr/0001-python-only-drop-rust.md) | Python-only; drop the Rust migration | Accepted |
| ADR-REPO-0002 | [0002-drop-bazel-uv-workspace.md](adr/0002-drop-bazel-uv-workspace.md) | Drop Bazel; `uv` workspace + import-linter | Accepted |
| ADR-REPO-0003 | [0003-subsystem-app-over-typed-bus.md](adr/0003-subsystem-app-over-typed-bus.md) | Subsystem-app model over a typed message bus | Accepted |
| ADR-REPO-0004 | [0004-onnx-detector-backend.md](adr/0004-onnx-detector-backend.md) | ONNX frozen-artifact detector behind a swappable backend | Accepted |
| ADR-REPO-0005 | [0005-pure-core-thin-shell.md](adr/0005-pure-core-thin-shell.md) | Pure-core + thin-shell apps; `Result` over exceptions | Accepted |
| ADR-REPO-0006 | [0006-iss-attached-reliability-posture.md](adr/0006-iss-attached-reliability-posture.md) | ISS-attached reliability posture | Accepted |
| ADR-REPO-0007 | [0007-raw-mosaic-sensor-ingest.md](adr/0007-raw-mosaic-sensor-ingest.md) | Raw-mosaic sensor ingest contract | Accepted |
| ADR-REPO-0008 | [0008-closed-loop-gimbal-pointing.md](adr/0008-closed-loop-gimbal-pointing.md) | Closed-loop gimbal pointing | Accepted |
| ADR-REPO-0009 | [0009-iss-link-transport-command-ingress.md](adr/0009-iss-link-transport-command-ingress.md) | ISS link transport + authenticated command ingress | Accepted |
| ADR-REPO-0010 | [0010-validation-configuration-matrix.md](adr/0010-validation-configuration-matrix.md) | Validation as a configuration matrix with a VCRM spine | Accepted |

## Notes

Legacy files keep their historical filenames and H1 titles. The `ADR-REPO-NNNN`
identifier is the canonical ID for indexes and supersession links.
