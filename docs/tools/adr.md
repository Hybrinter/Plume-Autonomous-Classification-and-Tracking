# Tools package ADRs

**Scope:** `ADR-TOOLS`
**Directory:** [`adr/`](adr/)

Package-local decisions for `packages/tools`. New records use files under
`adr/` named `NNNN-short-title.md` with title `# ADR-TOOLS-NNNN: ...`.

## Predecessor records

| ID | Decision |
| --- | --- |
| ADR-REPO-0001 | Python-only; drop the Rust migration |
| ADR-REPO-0002 | Drop Bazel; `uv` workspace + import-linter |

## Records

| ID | File | Decision | Status |
| --- | --- | --- | --- |
| ADR-TOOLS-0001 | [0001-tools-model-package.md](adr/0001-tools-model-package.md) | One `tools.inference` package for train, export, and accept | Accepted |
| ADR-TOOLS-0002 | [0002-plain-torch-train-loop.md](adr/0002-plain-torch-train-loop.md) | Plain torch train loop, no Lightning | Accepted |
| ADR-TOOLS-0003 | [0003-two-onnx-artifacts.md](adr/0003-two-onnx-artifacts.md) | Two frozen ONNX artifacts with a classifier filter | Accepted |
| ADR-TOOLS-0004 | [0004-dataset-out-of-git.md](adr/0004-dataset-out-of-git.md) | Dataset out of git; Zenodo fetch, not Git LFS | Accepted |
| ADR-TOOLS-0005 | [0005-tools-cli-and-inference-package.md](adr/0005-tools-cli-and-inference-package.md) | Use a nested tools CLI and inference package | Accepted |
| ADR-TOOLS-0006 | [0006-local-run-catalog.md](adr/0006-local-run-catalog.md) | Local filesystem run catalog; no tracking SaaS | Accepted |
| ADR-TOOLS-0007 | [0007-int8-qdq-ptq.md](adr/0007-int8-qdq-ptq.md) | INT8 is post-training QDQ PTQ; I/O stays float32 | Accepted |
| ADR-TOOLS-0008 | [0008-torch-required-tools-dep.md](adr/0008-torch-required-tools-dep.md) | Torch and torchvision are required tools deps | Accepted |
| ADR-TOOLS-0009 | [0009-unique-run-directories.md](adr/0009-unique-run-directories.md) | Unique run directories; refuse overwrite | Accepted |
