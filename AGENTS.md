# AGENTS.md

For cross-cutting code patterns and subsystem invariants, see `CLAUDE.md` and the per-subsystem
`CONTEXT.md` files. This file captures operational notes for agents working in this repo.

## Cursor Cloud specific instructions

This is a Python 3.14+ `uv` workspace (virtual root; members under `packages/`). Dependencies are
refreshed automatically on VM startup by the environment update script (`uv sync --extra dev`), so
you normally do not need to install anything yourself.

- Toolchain: `uv` is used for everything; the pinned interpreter is CPython 3.14 (managed by `uv`,
  not the system `python3` which is 3.12). Always invoke tools through `uv run ...` so they use the
  workspace `.venv`.
- Standard commands are documented in `README.md` ("Run the gates") and `.github/workflows/ci.yml`.
  The full gate set is: `uv run ruff check packages scripts`, `uv run ruff format --check packages
  scripts`, `uv run mypy packages scripts`, `uv run lint-imports`, `uv run python
  scripts/check_vcrm.py`, and `uv run pytest -m "not e2e"`.
- There is no GUI and no long-running service to start for development. The product runs
  in-process: the primary end-to-end path is the GSE harness stepping the real flight apps over sim
  (or real-loopback) drivers. Nothing needs `docker`, a database, or external daemons.
- Run the documented end-to-end scenarios (the "application") from the repo root, e.g.:
  `uv run python -c "from gse.scenario import load_scenario; from gse.orchestrator import
  run_scenario; s=load_scenario('scenarios/closed_loop_pointing.toml');
  print(run_scenario(s, f'profiles/{s.profile}.toml'))"`. Scenarios live in `scenarios/*.toml`;
  profiles in `profiles/*.toml`. `sil-link-real` scenarios start an in-process TCP/UDP CCSDS
  station emulator over loopback — no external setup required.
- The `tools.analysis` CLI (`uv run python -m tools.analysis list` / `run <suite|scenario> --out
  DIR`) captures deterministic SIL runs and writes Parquet/CSV + matplotlib report bundles; useful
  for observability but not required for scenario pass/fail.
- Design studies live under `packages/analysis/` (`uv run python -m analysis.studies.<study> ...`).
  They depend on flight and sim. Do not confuse them with `tools.analysis`.
- `flight.core.main` is the production entry point for real payload hardware (PySpin, pyserial,
  onnxruntime, camera/gimbal, HMAC key file). It is not runnable in this environment and is not
  needed for dev/CI; use the SIL/GSE paths instead.
- The `pytest` `e2e` marker is defined but currently unused (`-m "e2e"` selects zero tests); the
  full suite runs under `-m "not e2e"`.
