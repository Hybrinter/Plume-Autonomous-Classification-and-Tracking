# Architecture Decision Records

Short records of the load-bearing decisions behind the PACT flight-software restructure
(branch `fsw-restructure`, 2026-05-30 onward). Full design rationale:
`docs/superpowers/specs/2026-05-30-pact-iss-payload-fsw-structure-design.md`.

| ADR | Decision | Status |
|-----|----------|--------|
| [0001](0001-python-only-drop-rust.md) | Python-only; drop the Rust migration | Accepted |
| [0002](0002-drop-bazel-uv-workspace.md) | Drop Bazel; `uv` workspace + import-linter | Accepted |
| [0003](0003-subsystem-app-over-typed-bus.md) | Subsystem-app model over a typed message bus | Accepted |
| [0004](0004-onnx-detector-backend.md) | ONNX frozen-artifact detector behind a swappable backend | Accepted |
| [0005](0005-pure-core-thin-shell.md) | Pure-core + thin-shell apps; `Result` over exceptions | Accepted |
| [0006](0006-iss-attached-reliability-posture.md) | ISS-attached reliability posture (fail-safe / ground-recoverable) | Accepted |
