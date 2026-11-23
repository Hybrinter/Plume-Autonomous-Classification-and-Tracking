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
| ADR-TOOLS-0005 | [0005-tools-cli-and-inference-package.md](adr/0005-tools-cli-and-inference-package.md) | Use a nested tools CLI and inference package | Superseded |
| ADR-TOOLS-0006 | [0006-local-run-catalog.md](adr/0006-local-run-catalog.md) | Local filesystem run catalog; no tracking SaaS | Superseded |
| ADR-TOOLS-0007 | [0007-int8-qdq-ptq.md](adr/0007-int8-qdq-ptq.md) | INT8 is post-training QDQ PTQ; I/O stays float32 | Accepted |
| ADR-TOOLS-0008 | [0008-torch-required-tools-dep.md](adr/0008-torch-required-tools-dep.md) | Torch and torchvision are required tools deps | Accepted |
| ADR-TOOLS-0009 | [0009-unique-run-directories.md](adr/0009-unique-run-directories.md) | Unique run directories; refuse overwrite | Superseded |
| ADR-TOOLS-0010 | [0010-local-cartesian-sweep.md](adr/0010-local-cartesian-sweep.md) | Local cartesian sweep over the run catalog | Accepted |
| ADR-TOOLS-0011 | [0011-windows-cuda-torch-index.md](adr/0011-windows-cuda-torch-index.md) | Windows tools installs CUDA torch wheels | Accepted |
| ADR-TOOLS-0012 | [0012-ci-omits-tools-torch.md](adr/0012-ci-omits-tools-torch.md) | Lean CI shards omit pact-tools | Accepted |
| ADR-TOOLS-0013 | [0013-ml-models-package.md](adr/0013-ml-models-package.md) | One `tools.ml_models` package for train, export, and accept | Accepted |
| ADR-TOOLS-0014 | [0014-finished-datasets-per-source.md](adr/0014-finished-datasets-per-source.md) | Finished datasets per source | Superseded |
| ADR-TOOLS-0015 | [0015-gsd-film-conditioning.md](adr/0015-gsd-film-conditioning.md) | GSD FiLM conditioning in tools training | Superseded |
| ADR-TOOLS-0016 | [0016-two-input-export-contract.md](adr/0016-two-input-export-contract.md) | Two-input ONNX export contract and pair gates | Accepted |
| ADR-TOOLS-0017 | [0017-single-source-unit-tile-workflow.md](adr/0017-single-source-unit-tile-workflow.md) | Single-source unit-tile workflow | Accepted |
| ADR-TOOLS-0018 | [0018-evidence-first-phased-ml-analysis.md](adr/0018-evidence-first-phased-ml-analysis.md) | Evidence-first phased ML analysis | Accepted |
